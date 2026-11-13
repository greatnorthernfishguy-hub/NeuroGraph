# ---- Changelog ----
# [2026-11-13] DudeMan CC (Claude Fable 5.2, laptop) -- native tick piece 1 (punch #1221), parity before the engine edit
# What: runs ng_tract.PredictionBook's confirm / expire beside the engine's own Python (`_evaluate_predictions`,
#       `_cleanup_predictions`, `_on_prediction_confirmed`, `_on_prediction_error`, `_surprise_exploration`,
#       `_log_novel_sequence`) on identical twin graphs and compares the checkpoints byte for byte.
# Why:  plan docs/superpowers/plans/2026-11-13-native-tick-m1-prediction-miss-path.md §3; `neuro_foundation.py` is
#       protected, so until Josh's "proceed" the engine is NOT edited: the candidate side is the same engine file
#       imported a second time with those two methods replaced IN THIS TEST by a shim over the book.
# How:  the shim copies the Python state into the book before each pass and back after it (the engine still owns the
#       state on this branch), so every effect of a confirmation and of a miss is the book's; the host half (the tally
#       touch, the rails, create_synapse, event emission) is the engine's own code, as it stays in piece 1.
#       Needs an ng_tract with PredictionBook (ng-tract-rs branch cc-laptop-native-tick-m1-20261113); skipped otherwise.
# -------------------
import importlib.util
import os
import random
import sys
from collections import deque

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


def _load_twin(name):
    spec = importlib.util.spec_from_file_location(name, NF.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SHIM = _load_twin("nf_native_tick_m1_shim")
CHECKS = {"windows": 0, "confirm": 0, "miss": 0, "recv_miss": 0, "host_miss": 0}


def _book(g):
    b = g.__dict__.get("_pb_shim")
    if b is None:
        b = g.__dict__["_pb_shim"] = SHIM.ng_tract.PredictionBook()
    return b


def _sync_in(g, b):
    b.spike_clear()
    for nid, dq in g._recent_spikes.items():
        b.spike_set(nid, list(dq), dq.maxlen)
    b.hist_clear()
    for sid, dq in g._synapse_confirmation_history.items():
        b.hist_set(sid, list(dq))
    b.clear()
    for pid, p in g.active_predictions.items():
        recv = None
        if p.contributors is not None:
            recv = ([(sid, pre, w) for sid, (pre, w) in p.contributors_raw.items()], p.drive, p.rho)
        b.insert(pid, p.source_node_id, p.target_node_id, p.strength, p.confidence, p.created_at, p.expires_at,
                 p.chain_depth, p.via_hyperedge, p.pre_charge_applied, recv)
        if recv is not None:
            assert b.contributors(pid) == [tuple(x) for x in p.contributors], "shares differ from the engine's"
    b.outcomes_clear()
    b.novel_clear()
    for e in g._novel_sequence_log:
        b.novel_push_restored(e["source"], list(e["firing_nodes"]), e["n_firing"], e["timestep"])
    b.set_counters(g._total_predictions_made, g._total_predictions_confirmed, g._total_predictions_errors,
                   g._total_novel_sequences)


def _sync_out(g, b, objs):
    for pid in [pid for pid in g.active_predictions if pid not in b]:
        del g.active_predictions[pid]
    g._synapse_confirmation_history = {sid: deque(v, maxlen=100) for sid, v in b.hist_items()}
    for (ids, _nums, _rest), confirmed, resolved_at, actual, n_actual in b.outcome_items():
        g._prediction_outcomes.append(SHIM.PredictionOutcome(
            prediction=objs[ids[0]], confirmed=confirmed, resolved_at=resolved_at, actual_firing_nodes=actual,
            n_actual_firing=n_actual))
    g._novel_sequence_log = [{"source": s, "firing_nodes": f, "n_firing": n, "timestep": t}
                             for s, f, n, t in b.novel_items()]
    _made, g._total_predictions_confirmed, g._total_predictions_errors, g._total_novel_sequences = b.counters()
    s10, s13 = b.take_reward_events()
    if s10:
        g.__dict__.setdefault("_s10_events", []).extend(s10)
    if s13:
        g.__dict__.setdefault("_s13_events", []).extend(s13)


def _flags(g):
    return g._stab_s10_rpe() is not None, g._stab_s13_per_source() is not None


def _shim_evaluate(self, fired_set):
    b = _book(self)
    _sync_in(self, b)
    objs = dict(self.active_predictions)
    s10, s13 = _flags(self)
    rows = b.confirm(self.timestep, list(fired_set), self.synapses, self.nodes,
                     self.config["prediction_confirm_bonus"], self.config["three_factor_enabled"], s10, s13, True)
    CHECKS["confirm"] += len(rows)
    _sync_out(self, b, objs)
    for source, target, strength, confidence, delay in rows:
        self._emit("prediction_confirmed", source=source, target=target, strength=strength, confidence=confidence,
                   delay=delay, timestep=self.timestep)


def _python_recent_fired(g):
    out = set()
    for nid, spikes in g._recent_spikes.items():
        for t in spikes:
            if g.timestep - t <= g.config["prediction_window"]:
                out.add(nid)
                break
    return out


def _host_explore(self, pred, b):
    """The engine's exploration step for one source-side miss (neuro_foundation.py :5953-6038): unchanged code."""
    cfg = self.config
    expected_id = pred.target_node_id
    _tally = bool(cfg.get("sprout_tally_enabled", False))
    _win = _tally and getattr(self, "_surprise_window_t", None) == self.timestep
    if cfg.get("surprise_slot_enabled", False):
        alternative = self._surprise_slot_alternatives(pred)
    else:
        alternative = None if _win else set(b.window_ids()) - {expected_id}
    sprouted = 0
    if _tally:
        sprouted = self._surprise_exploration_tally(pred, alternative)
    else:
        raise AssertionError("the no-tally source-side path is not covered by this shim (creation order: see plan)")
    if sprouted > 0 and self._event_handlers.get("surprise_sprouted"):
        alts = [a for a in b.window_ids() if a != expected_id] if alternative is None else sorted(alternative)
        self._emit("surprise_sprouted", source=pred.source_node_id, expected=expected_id, alternatives=alts,
                   count=sprouted, timestep=self.timestep)
    return None if alternative is None else sorted(alternative)


def _shim_cleanup(self):
    b = _book(self)
    _sync_in(self, b)
    objs = dict(self.active_predictions)
    cfg = self.config
    s10, s13 = _flags(self)
    heard = [bool(self._event_handlers.get(k)) for k in ("prediction_error", "surprise_sprouted", "novel_sequence")]
    dp_p = self.__dict__.get("_diffpc_p")
    dps = self.__dict__.get("_diffpc") if dp_p is not None else None
    n = b.expire_begin(self.timestep, cfg["prediction_window"], cfg["prediction_error_penalty"],
                       cfg["three_factor_enabled"], cfg["co_activation_window"], cfg["he_salience_max"],
                       cfg["prediction_threshold"], s10, s13, cfg.get("surprise_reward_scaling", 0.5),
                       (bool(dp_p[3]) if dps is not None else None), any(heard))
    want = _python_recent_fired(self)
    assert b.window_count() == len(want) and set(b.window_ids()) == want, "the fired window differs"
    CHECKS["windows"] += 1
    self._surprise_window_t = None
    if (n and cfg.get("sprout_tally_enabled", False) and not cfg.get("surprise_slot_enabled", False) and not heard[0]):
        _sw = SHIM._cofire_surprise_native(self.synapses)
        if _sw is not None and SHIM._cofire_tally_native(self.synapses) is not None:
            _sw[0](b.window_ids(self.nodes))
            self._surprise_window_t = self.timestep
    while True:
        m = b.expire_run(self.synapses, self.nodes, dps)
        if m is None:
            break
        pid, source, target, strength, confidence, _created, _depth, is_recv = m
        pred = objs[pid]
        CHECKS["host_miss"] += 1
        if heard[0]:
            self._emit("prediction_error", source=source, expected_target=target, strength=strength,
                       confidence=confidence, actual_fired=b.window_ids(), timestep=self.timestep)
        alts = None if is_recv else _host_explore(self, pred, b)
        novel, reward = b.expire_finish(self.synapses, alts)
        if novel and heard[2]:
            firing = [a for a in b.window_ids() if a != target] if alts is None else alts
            self._emit("novel_sequence", source=source, firing_nodes=firing, timestep=self.timestep)
        if reward is not None:
            self.inject_reward(reward)
    CHECKS["miss"] += n
    CHECKS["recv_miss"] += sum(1 for pid in objs if pid not in b and objs[pid].contributors is not None
                               and self.timestep > objs[pid].expires_at)
    _sync_out(self, b, objs)


SHIM.Graph._evaluate_predictions = _shim_evaluate
SHIM.Graph._cleanup_predictions = _shim_cleanup


def _reset():
    for k in CHECKS:
        CHECKS[k] = 0


# ---------------------------------------------------------------------------
# (1) receiver on (S18), with and without the corrected DiffPC (S19): the S18 whole-run twin, engine vs book
# ---------------------------------------------------------------------------

@S18.needs_recv
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("s19", [False, True])
def test_receiver_whole_run_book_equals_engine(seed, s19):
    _reset()
    start = H2.built(seed)
    t = S.Pair(S18.restore_as(NF, "native", start), S18.restore_as(SHIM, "native", start), seed)
    extra = {"diffpc_receiver_enabled": True} if s19 else {}
    t.op("keys", lambda g: g.config.update(dict(S18.ARM, **S18.RPE, **S18.ON, **extra)))
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert CHECKS["windows"] > 0 and CHECKS["confirm"] + CHECKS["miss"] > 0, CHECKS      # not vacuous
    assert S18._state_rows(t.a.g) == S18._state_rows(t.b.g)
    assert S18.ckpt(t.a.g) == S18.ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) source-side with the co-firing tally (the live daemon's path): the #1050 whole-run workload, engine vs book
# ---------------------------------------------------------------------------

def _restore(mod, b):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = mod.Graph(native_node_store=True)
        g.restore(p)
        return g


def _run(mod, start, act, seed, cfg, heard=False):
    g = _restore(mod, start)
    g.config.update(cfg)
    log = T1050._heard(g) if heard else []
    out = P1.seeded(seed, lambda rng: act(g, rng), uuid_stream=1)
    return P1.ckpt_bytes(g), out, log


def _find_first(self, pre_id, post_id):
    """Graph._find_synapse under the book's fixed rule for a pair with two synapses (the lowest row). The engine's own
    walk of a set of synapse ids picks by str hash order there, which differs from process to process (found
    2026-11-13: 22 of 876 pairs in the P1 graph have two synapses), so no fixed rule can equal it; both sides of the
    twin get this rule, and everything else must then be byte-equal."""
    sid = self.synapses.find_first(pre_id, post_id)
    return self.synapses.get(sid) if sid is not None else None


@T1050.needs_window
@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("heard", [False, True])
def test_tally_whole_run_book_equals_engine(seed, heard, monkeypatch):
    _reset()
    monkeypatch.setattr(NF.Graph, "_find_synapse", _find_first)
    monkeypatch.setattr(SHIM.Graph, "_find_synapse", _find_first)
    start = P1.built_bytes(seed)
    cfg = dict(T1050.TALLY, **T1050.DISUSE, prediction_threshold=0.5)   # the workload's weights predict
    act = T1050.tally_workload(30)
    a, (la, _), ea = _run(NF, start, act, seed, cfg, heard)
    b, (lb, _), eb = _run(SHIM, start, act, seed, cfg, heard)
    assert CHECKS["windows"] > 0 and CHECKS["miss"] > 0, CHECKS
    assert ea == eb and (len(ea) > 0) == heard                 # every event, with its ids, when heard
    P1.compare(la, lb, [], [], a, b)


# ---------------------------------------------------------------------------
# (3) the registration calls the engine edit will use (not reached through the shim): merge and the pending index
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
