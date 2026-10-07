# ---- Changelog ----
# [2026-10-07] Claude Opus 5.5 (lane guardian-1051) — #1051 reconciled synapse gate: replays + concurrency proofs
# What: (A) replays of the real incidents through SaveGate with a scripted removal stream: 2026-10-06 staged collapse
#   (punch-list numbers 64,327 -> 34,652 -> 21,618, and the log-derived 55,260 -> 28,691 -> 21,274), with and without
#   matching logged removals; 2026-10-04 unexplained 86% loss; a slow unexplained walk; a sleep-sized explained
#   clearance; empty/near-empty graphs; restart mid-history; an out-of-band manifest (promoted quarantine); the save
#   history extracted from the CC daemon log (tests/fixtures/guardian_1051_daemon_log_saves.json) replayed as is.
#   (B) the real NeuroGraphMemory + real Graph: default-off parity, removals racing saves (exact residual 0), a save
#   requested mid-sleep_cycle, removals between the detached capture and the permit, a corrupt-restore boot, restart.
# Why: punch list #1051 proof bar (lane brief 2026-10-07).
# How: NG_GUARDIAN_RECONCILE set per test with monkeypatch; tmp workspaces; no live checkpoint is read or written.
# -------------------
"""#1051: the checkpoint guardian tells plasticity (logged removals) from damage (unexplained loss)."""

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import checkpoint_guardian as cg
from checkpoint_guardian import (
    RemovalLedger,
    SaveGate,
    evaluate_save_health,
    evaluate_synapse_reconciliation,
    guard_history_path_for,
    read_guard_history,
    write_manifest,
)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "guardian_1051_daemon_log_saves.json")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("NG_GUARDIAN_RECONCILE", "1")
    for k in list(os.environ):
        if k.startswith("NG_GUARDIAN_") and k != "NG_GUARDIAN_RECONCILE":
            monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ckpt():
    d = tempfile.mkdtemp(prefix="guardian1051_")
    yield os.path.join(d, "main.msgpack")
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- scripted host (SaveGate level)

class FakeGraph:
    """Only what the ledger touches: the public event API and a synapse count."""

    def __init__(self, synapses=0):
        self._h = {}
        self.synapses = range(synapses)

    def register_event_handler(self, ev, cb):
        self._h.setdefault(ev, []).append(cb)

    def emit(self, ev, **kw):
        for cb in self._h.get(ev, []):
            cb(**kw)


class Sim:
    """One host process: what openclaw_hook.save() does around the gate, minus the graph I/O."""

    def __init__(self, ckpt, nodes=11281, boot_synapses=0):
        self.ckpt = ckpt
        self.nodes = nodes
        self.g = FakeGraph(boot_synapses)
        self.gate = SaveGate(ckpt)
        self.gate.record_restore("ok" if os.path.exists(ckpt) else "no_file", nodes)
        self.ledger = self.gate.attach_removal_ledger(self.g)
        if not os.path.exists(ckpt):
            open(ckpt, "wb").close()

    def prune(self, n):
        if n:
            self.g.emit("pruned", count=n, timestep=0)

    def sleep(self, n):
        self.prune(n)            # sleep_cycle's own _prune_synapses emits "pruned" ...
        self.g.emit("sleep_cycle", pruned=n, nodes_collected=0)   # ... then the attribution event

    def save(self, synapses):
        counts = {"nodes": self.nodes, "guardian_nodes": self.nodes, "synapses": synapses, "hyperedges": 0,
                  "timestep": 0, "vdb_count": 0}
        if self.ledger is not None:
            counts["removals"] = self.ledger.snapshot()
        kw = {"removals": counts["removals"]} if "removals" in counts else {}
        ok, reason = self.gate.permit(self.nodes, live_synapses=synapses, live_hyperedges=0, **kw)
        recon = self.gate._pending_recon
        if ok:
            write_manifest(self.ckpt, {k: v for k, v in counts.items() if k != "removals"})
            time.sleep(0.001)   # distinct saved_at stamps
            self.gate.record_accepted(counts)
        return ok, reason, recon


def _feed(sim, seq):
    """seq: [(synapses_at_save, removals_logged_before_it)] -> list of (ok, reason, recon)."""
    out = []
    for syn, pruned in seq:
        sim.prune(pruned)
        out.append(sim.save(syn))
    return out


# 10 accepted saves from the real log before the 10-06 cascade (segment booted at line 634361), approx counts and the
# removals logged between them -- then the punch-list's last accepted save, 64,327.
PRE_1006 = [(65213, 9), (65208, 75), (65483, 86), (65881, 87), (66308, 1), (66616, 281), (66616, 0), (65771, 1474),
            (65881, 173), (66042, 133), (64327, 1800)]


def _pre_1006(sim):
    res = _feed(sim, PRE_1006)
    assert all(ok for ok, _, _ in res), [r for _, r, _ in res]
    return res


# --------------------------------------------------------------------------- (A) replays

def test_1006_staged_collapse_without_removal_events_refused_at_first_step(on, ckpt, caplog):
    sim = Sim(ckpt)
    _pre_1006(sim)
    with caplog.at_level(logging.INFO):
        ok, reason, r = sim.save(34652)              # nothing logged: 29,675 synapses vanished
    assert not ok and "unexplained synapse loss" in reason
    assert r["explained"] == 0 and r["unexplained"] > r["tolerance"]
    ok2, reason2, _ = sim.save(21618)                # the second step: still measured against the median
    assert not ok2 and "unexplained" in reason2
    assert read_guard_history(ckpt)[-1]["synapses"] == 64327   # refused saves never advance the reference


def test_1006_staged_collapse_with_matching_events_passes_and_alarms(on, ckpt, caplog):
    sim = Sim(ckpt)
    _pre_1006(sim)
    with caplog.at_level(logging.INFO, logger="checkpoint_guardian"):
        sim.prune(64327 - 34652)
        ok, reason, r = sim.save(34652)
        assert ok and r["permit"] and r["alarm"]
        sim.prune(34652 - 21618)
        ok2, _, r2 = sim.save(21618)
    assert ok2 and r2["alarm"]
    churn = [x for x in caplog.records if "UNUSUAL CHURN" in x.getMessage()]
    assert len(churn) == 2 and all(x.levelno == logging.WARNING for x in churn)
    assert "does not block" in churn[0].getMessage()
    h = read_guard_history(ckpt)
    assert [e["synapses"] for e in h[-2:]] == [34652, 21618]
    assert [e["explained_synapses"] for e in h[-2:]] == [29675, 13034]


def test_1006_log_derived_sequence(on, ckpt):
    """The daemon log puts the 10:03 / 10:11 / 10:23 saves at ~55,260 / ~28,691 / ~21,274 (last prune line before each
    save) with 26,976 and 7,871 synapses pruned in between. Logged: accepted + alarm. Not logged: refused at once."""
    seq = [(64029, 298), (55260, 8970)]
    sim = Sim(ckpt)
    _pre_1006(sim)
    assert all(ok for ok, _, _ in _feed(sim, seq))
    (ok1, _, r1), (ok2, _, r2) = _feed(sim, [(28691, 26976), (21274, 7871)])
    assert ok1 and ok2 and r1["alarm"] and r2["alarm"]

    shutil.rmtree(os.path.dirname(ckpt)); os.makedirs(os.path.dirname(ckpt))
    sim = Sim(ckpt)
    _pre_1006(sim)
    _feed(sim, seq)
    ok, reason, _ = sim.save(28691)
    assert not ok and "unexplained" in reason


def test_1004_unexplained_86pct_loss_refused(on, ckpt):
    """2026-10-04 12:56: synapses 231,934 -> 31,925, nodes 9,347 -> 9,154: removed by remove-false-wants
    (host remove_node cascades), which the engine does not log as plasticity."""
    sim = Sim(ckpt, nodes=9347)
    _feed(sim, [(231934 - 300 * i, 250) for i in range(10, -1, -1)])
    sim.nodes = 9154
    ok, reason, r = sim.save(31925)
    assert not ok and "unexplained synapse loss" in reason
    assert r["unexplained"] > 190000


def test_slow_unexplained_walk_is_stopped(on, ckpt):
    """A collapse in small unexplained steps (each one inside the per-step tolerance) is refused once the steps add up
    against the median; the 50%-of-last-save rule would have walked all the way down."""
    sim = Sim(ckpt)
    _pre_1006(sim)
    live, steps, refused_at = 64327, 0, None
    while live > 21618:
        live = int(live * 0.985)                     # -1.5% per save, nothing logged
        steps += 1
        ok, _, r = sim.save(live)
        assert evaluate_save_health(11281, 11281, live, int(live / 0.985) + 1)[0]   # legacy: every step passes
        if not ok:
            refused_at = steps
            break
    assert refused_at is not None and refused_at <= 4
    assert 64327 - live < 0.07 * 64327               # total unexplained walk bounded by the tolerance


def test_sleep_sized_clearance_explained_passes_with_alarm(on, ckpt, caplog):
    """Spec 2026-10-06 §5 / D7: the first clearance removes ~half the graph (15,957 of 32,190) in one sleep."""
    sim = Sim(ckpt)
    _feed(sim, [(32190 - 40 * i, 60) for i in range(10, -1, -1)])
    with caplog.at_level(logging.INFO, logger="checkpoint_guardian"):
        sim.sleep(15957)
        ok, _, r = sim.save(32190 - 15957 + 120)     # + a little wake sprouting
    assert ok and r["alarm"] and r["sleep_cycles"] == 1 and r["sleep_synapses"] == 15957
    assert r["explained"] == 15957                   # the sleep_cycle event did NOT double count
    info = [x.getMessage() for x in caplog.records if "reconcile:" in x.getMessage()]
    assert info and "1 sleep_cycle(s) removed 15957" in info[-1]
    # the median re-centres: three quiet saves later the next normal save raises no alarm
    for i in range(1, 6):
        ok, _, r = sim.save(16353 + 30 * i)
    assert ok and not r["alarm"]


def test_empty_and_near_empty_graphs_refused(on, ckpt):
    sim = Sim(ckpt)
    _pre_1006(sim)
    sim.prune(64327)
    ok, reason, _ = sim.save(0)                      # everything "explained" -- still the clobber shape
    assert not ok and "less than 10%" in reason
    sim.nodes = 3                                    # an empty graph next to a real mind: the node floor, unchanged
    ok, reason, _ = sim.save(64327)
    assert not ok and "absolute floor" in reason


def test_restart_in_the_middle_of_a_history(on, ckpt):
    """Same verdicts whether or not the process restarted between saves; history survives; ledger restarts at 0."""
    a = Sim(ckpt)
    _feed(a, PRE_1006[:6])
    b = Sim(ckpt, boot_synapses=PRE_1006[5][0])      # restart: restored == the last accepted save
    assert b.gate._ledger_baseline is None
    rest_b = _feed(b, PRE_1006[6:])
    ok_b, _, rb = b.save(34652)

    d2 = tempfile.mkdtemp(prefix="guardian1051_ref_")
    try:
        c = Sim(os.path.join(d2, "main.msgpack"))
        _feed(c, PRE_1006)
        ok_c, _, rc = c.save(34652)
    finally:
        shutil.rmtree(d2, ignore_errors=True)
    assert all(ok for ok, _, _ in rest_b)
    assert (ok_b, rb["reference"], rb["tolerance"]) == (ok_c, rc["reference"], rc["tolerance"])
    assert not ok_b
    assert len(read_guard_history(ckpt)) == 10


def test_restart_with_lost_removals_is_not_unexplained(on, ckpt):
    a = Sim(ckpt)
    _feed(a, PRE_1006)
    a.prune(30000)                                   # removed, never saved: dies with the process
    b = Sim(ckpt, boot_synapses=64327)               # restores the last accepted save
    ok, _, r = b.save(64327 + 50)
    assert ok and r["explained"] == 0 and r["unexplained"] <= 0


def test_boot_mismatch_is_warned_not_laundered(on, ckpt, caplog):
    _feed(Sim(ckpt), PRE_1006)
    with caplog.at_level(logging.WARNING, logger="checkpoint_guardian"):
        b = Sim(ckpt, boot_synapses=20000)           # the restore came back short and nothing explains it
    assert any("NOT explained" in x.getMessage() for x in caplog.records)
    ok, reason, _ = b.save(20000)
    assert not ok and "unexplained" in reason


def test_out_of_band_manifest_reseeds_history(on, ckpt, caplog):
    """A quarantine promoted by hand with a truthful write_manifest (2026-10-04) becomes the new reference."""
    sim = Sim(ckpt)
    _pre_1006(sim)
    write_manifest(ckpt, {"nodes": 11281, "guardian_nodes": 11281, "synapses": 31925, "hyperedges": 0})
    with caplog.at_level(logging.WARNING, logger="checkpoint_guardian"):
        ok, _, r = sim.save(31925 + 20)
    assert ok and any("History re-seeded" in x.getMessage() for x in caplog.records)
    h = read_guard_history(ckpt)
    assert [e["source"] for e in h] == ["out_of_band", "save"] and h[0]["synapses"] == 31925
    assert r["tolerance_basis"].startswith("bootstrap")


def test_failed_or_refused_save_never_advances(on, ckpt):
    sim = Sim(ckpt)
    _pre_1006(sim)
    before = read_guard_history(ckpt)
    sim.prune(500)
    counts_ok, _ = sim.gate.permit(11281, live_synapses=63827, live_hyperedges=0, removals=sim.ledger.snapshot())
    assert counts_ok                                 # permitted, but the write "failed": record_accepted never called
    assert read_guard_history(ckpt) == before
    ok, _, r = sim.save(63827)                       # the next save still reconciles against 64,327 with all 500
    assert ok and r["explained"] == 500 and r["last"] == 64327


def test_daemon_log_save_history_replayed(on):
    """Every save in the CC daemon log 2026-10-05 12:35 .. 2026-10-06 23:52 (11 process segments, 250 saves). With the
    removals the log shows, none is refused; the churn alarm fires on the two cascades only (10-05 evening, 10-06
    morning). With the removals hidden, both cascades are refused at their first step."""
    segs = json.load(open(FIXTURE))
    assert sum(len(s["saves"]) for s in segs) >= 240
    alarms, hidden_refusals = [], []
    for s in segs:
        for hide in (False, True):
            d = tempfile.mkdtemp(prefix="guardian1051_log_")
            try:
                sim = Sim(os.path.join(d, "main.msgpack"))
                for i, sv in enumerate(s["saves"]):
                    if not hide:
                        sim.prune(sv["pruned_since_prev"])
                    ok, reason, r = sim.save(sv["synapses"])
                    if not hide:
                        assert ok, (s["boot_line"], i, reason)
                        if r and r.get("alarm"):
                            alarms.append((s["boot_line"], i))
                    elif not ok:
                        hidden_refusals.append((s["boot_line"], i))
                        break
            finally:
                shutil.rmtree(d, ignore_errors=True)
    assert alarms == [(625899, 9), (625899, 10), (625899, 11), (636083, 2), (636083, 3)], alarms
    assert (625899, 9) in hidden_refusals and (636083, 1) in hidden_refusals


def test_default_off_is_the_legacy_gate(monkeypatch, ckpt):
    monkeypatch.delenv("NG_GUARDIAN_RECONCILE", raising=False)
    sim = Sim(ckpt)
    assert sim.ledger is None
    _feed(sim, PRE_1006)
    assert not os.path.exists(guard_history_path_for(ckpt))
    sim.prune(29675)                                 # logged, but nobody is listening
    ok, reason, recon = sim.save(31000)
    assert not ok and "below 50%" in reason and recon is None


# --------------------------------------------------------------------------- unit: ledger + pure function

def test_ledger_real_sleep_cycle_counts_once():
    from neuro_foundation import Graph
    g = Graph(config={"grace_period": 0, "inactivity_threshold": float("inf"), "orphan_node_grace_period": 10 ** 9})
    led = RemovalLedger().attach(g)
    ids = [g.create_node(node_id=f"n{i}").node_id for i in range(40)]
    syns = [g.create_synapse(ids[i], ids[(i + 1) % 40], weight=0.5).synapse_id for i in range(40)]
    for sid in syns[:13]:
        g.synapses[sid].weight = 0.0
    rec = g.sleep_cycle()
    s = led.snapshot()
    assert rec["pruned"] == 13 == s["synapses"] == s["sleep_synapses"] and s["sleep_cycles"] == 1
    assert len(g.synapses) == 27


def test_ledger_handler_error_never_raises_and_errs_strict(caplog):
    led = RemovalLedger()
    with caplog.at_level(logging.WARNING, logger="checkpoint_guardian"):
        led._on_pruned(count="not-a-number")
        led._on_pruned(count=object())
    s = led.snapshot()
    assert s["handler_errors"] == 2 and s["synapses"] == 0
    assert sum("UNDER-counted" in x.getMessage() for x in caplog.records) == 1


def test_median_reference_vs_a_sprout_burst():
    """Josh rejected a max-graph anchor: an overnight sprout burst would inflate it. The median follows the majority of
    the window: a burst in 3 of 10 saves does not move it; a burst that has lasted 6 of 10 saves has become the norm."""
    minority = [{"synapses": 30000, "explained_synapses": 50}] * 7 + [{"synapses": 60000, "explained_synapses": 50}] * 3
    r = evaluate_synapse_reconciliation(30500, minority, explained_synapses=0)
    assert r["permit"] and r["reference"] < 30500
    majority = [{"synapses": 30000, "explained_synapses": 50}] * 4 + [{"synapses": 60000, "explained_synapses": 50}] * 6
    r = evaluate_synapse_reconciliation(30500, majority, explained_synapses=0)
    assert not r["permit"] and r["reference"] > 59000


def test_not_applicable_without_history():
    r = evaluate_synapse_reconciliation(100, [], 0)
    assert r["applicable"] is False and r["permit"]


# --------------------------------------------------------------------------- (B) real NeuroGraphMemory + Graph

TEST_CONFIG = {"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}, "grace_period": 0,
               "inactivity_threshold": float("inf"), "orphan_node_grace_period": 10 ** 9}


@pytest.fixture
def ws():
    d = tempfile.mkdtemp(prefix="guardian1051_ws_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _boot(ws):
    from openclaw_hook import NeuroGraphMemory
    return NeuroGraphMemory(workspace_dir=ws, config=TEST_CONFIG)


def _wire(ng, n_nodes=150, n_syn=4000):
    g = ng.graph
    ids = [g.create_node(node_id=f"g1051-{i}").node_id for i in range(n_nodes)]
    k = 0
    for d in range(1, n_nodes):
        for i in range(n_nodes):
            if k >= n_syn:
                return
            g.create_synapse(ids[i], ids[(i + d) % n_nodes], weight=0.5)
            k += 1


def _sleep_remove(ng, k):
    """k synapses go below weight_threshold and the engine's own sleep_cycle prunes them (logged)."""
    g = ng.graph
    with g._step_lock:
        for sid in list(g.synapses.keys())[:k]:
            g.synapses[sid].weight = 0.0
        return g.sleep_cycle()["pruned"]


def _residuals(ws):
    h = read_guard_history(os.path.join(ws, "checkpoints", "main.msgpack"))
    return [p["synapses"] - c["explained_synapses"] - c["synapses"] for p, c in zip(h, h[1:])]


def test_real_default_off_parity(ws, monkeypatch):
    monkeypatch.delenv("NG_GUARDIAN_RECONCILE", raising=False)
    ng = _boot(ws)
    assert ng._removal_ledger is None
    _wire(ng)
    ng.save()
    assert "removals" not in ng._capture_checkpoint_state()["counts"]
    assert not os.path.exists(os.path.join(ws, "checkpoints", "main.msgpack.guard_history.json"))


def test_real_save_path_explained_vs_unexplained(on, ws):
    ng = _boot(ws)
    _wire(ng)
    assert ng.save().endswith("main.msgpack")
    for _ in range(4):
        _sleep_remove(ng, 40)
        assert ng.save().endswith("main.msgpack")
    assert _residuals(ws) == [0, 0, 0, 0]
    _sleep_remove(ng, 2000)                          # half the graph, logged -> accepted
    rec = ng.save(with_receipt=True)
    assert rec["outcome"] == "primary" and rec["accepted"]
    g = ng.graph
    for sid in list(g.synapses.keys())[:1500]:       # host removal, NOT logged -> damage
        g.remove_synapse(sid)
    out = ng.save()
    assert os.sep + "quarantine" + os.sep in out


def test_real_removals_racing_saves_count_exactly(on, ws):
    ng = _boot(ws)
    _wire(ng, n_nodes=200, n_syn=8000)
    ng.save()
    stop = threading.Event()
    removed = []

    def remover():
        while not stop.is_set() and len(ng.graph.synapses) > 3000:
            removed.append(_sleep_remove(ng, 7))
            time.sleep(0.0005)

    t = threading.Thread(target=remover)
    t.start()
    try:
        outs = []
        for i in range(25):
            outs.append(ng.save(with_receipt=bool(i % 2)))
    finally:
        stop.set()
        t.join()
    assert all((o["outcome"] == "primary") if isinstance(o, dict) else o.endswith("main.msgpack") for o in outs)
    res = _residuals(ws)
    assert res and all(x == 0 for x in res), res     # every logged removal in exactly one interval
    h = read_guard_history(os.path.join(ws, "checkpoints", "main.msgpack"))
    assert sum(e["explained_synapses"] for e in h[1:]) == h[0]["synapses"] - h[-1]["synapses"]
    assert sum(1 for e in h[1:] if e["explained_synapses"] > 0) >= 5   # removals really interleaved with the saves


def test_real_save_requested_mid_sleep_waits_for_the_whole_pass(on, ws):
    ng = _boot(ws)
    _wire(ng)
    ng.save()
    g = ng.graph
    box = {}

    def on_pruned(count=0, **_):                     # fires INSIDE sleep_cycle, under the step lock
        if "t" not in box:
            box["t"] = threading.Thread(target=lambda: box.setdefault("out", ng.save()))
            box["t"].start()
            time.sleep(0.2)
            box["done_during_sleep"] = "out" in box

    g.register_event_handler("pruned", on_pruned)
    n = _sleep_remove(ng, 300)
    box["t"].join()
    assert box["done_during_sleep"] is False
    h = read_guard_history(os.path.join(ws, "checkpoints", "main.msgpack"))
    assert h[-1]["explained_synapses"] == n == 300 and h[-1]["sleep_cycles"] == 1
    assert h[-1]["synapses"] == len(g.synapses)


def test_real_removal_after_detached_capture_counts_in_the_next_save(on, ws, monkeypatch):
    ng = _boot(ws)
    _wire(ng)
    ng.save()
    orig = ng._capture_checkpoint_state
    late = {"n": 0}

    def capture_then_remove():
        cap = orig()                                 # lock released here (#423 detached snapshot)
        if not late["n"]:
            late["n"] = _sleep_remove(ng, 250)       # before permit / write of THIS save
        return cap

    monkeypatch.setattr(ng, "_capture_checkpoint_state", capture_then_remove)
    _sleep_remove(ng, 100)
    ng.save()
    ng.save()
    h = read_guard_history(os.path.join(ws, "checkpoints", "main.msgpack"))
    assert [e["explained_synapses"] for e in h[-2:]] == [100, 250]
    assert _residuals(ws)[-2:] == [0, 0]


def test_real_corrupt_restore_still_provisional(on, ws):
    ng = _boot(ws)
    _wire(ng)
    primary = ng.save()
    good = open(primary, "rb").read()
    open(primary, "wb").write(good[: len(good) // 3])
    ng2 = _boot(ws)
    assert ng2._save_gate.provisional
    assert os.sep + "quarantine" + os.sep in ng2.save()


def test_real_restart_continues_history(on, ws):
    ng = _boot(ws)
    _wire(ng)
    ng.save()
    _sleep_remove(ng, 60)
    ng.save()
    _sleep_remove(ng, 999)                           # never saved: lost with the process
    ng2 = _boot(ws)
    assert ng2._removal_ledger.snapshot()["synapses"] == 0
    _sleep_remove(ng2, 70)
    assert ng2.save().endswith("main.msgpack")
    h = read_guard_history(os.path.join(ws, "checkpoints", "main.msgpack"))
    assert [e["source"] for e in h] == ["save", "save", "save"]
    assert _residuals(ws) == [0, 0]
