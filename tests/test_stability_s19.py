# ---- Changelog ----
# [2026-11-03] DudeMan CC (Claude Opus 5.5, laptop; #1202 follow-up, DiffPC corrected; Josh 2026-11-02 "DiffPC
#   shouldn't be turned off, it should be corrected" / "Yeah, this is good as well") — S19: DiffPC as the error half of
#   S18's receiver predictions
# What: (1) the switch absent / explicitly False, and count-only on, == the S18 line (15b1958's module) exactly (final
#       checkpoint BYTES); keys not in DEFAULT_CONFIG; bad params / no receiver log ERROR once and change nothing.
#       (2) the error on the arrival clock: confirmed / miss / surprise, explained drive = arrivals + hyperedge
#       injections, excitatory only, learned-from-surprise. (3) the |e| record, its one reset, the old body silent.
#       (4) the plateau: aimed at the predicted band, never at causation, top K, native == Python.
# Why:  docs/superpowers/specs/2026-11-02-diffpc-correction-design.md; plan 2026-11-02-diffpc-correction.md.
# How:  oracle via `git show 15b1958:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers;
#       small hand-built graphs for the mechanism.
# -------------------
"""#1202 follow-up, round S19: DiffPC corrected."""
import logging
import os
import random

import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S19_BASE_REV", "15b1958")
CMOD = H2._load(BASE_REV, "nf_s18_commit_s19")
S19_SWITCHES = ("diffpc_receiver_enabled", "diffpc_receiver_count_only")
S19_PARAMS = ("diffpc_receiver_eta", "diffpc_receiver_k", "diffpc_receiver_window")
ARM = S7B.S7B_ARM
RPE = {"three_factor_enabled": True, "reward_rpe_enabled": True}
RECV = {"prediction_receiver_enabled": True}
ON = dict(RECV, diffpc_receiver_enabled=True)
ckpt = S.ckpt
restore_as = S.restore_as


def _pair(kind, seed, base_cfg):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    t.op("keys", lambda g: g.config.update(dict(ARM, **base_cfg) if kind == "native" else dict(base_cfg)))
    return t


# ---------------------------------------------------------------------------
# (1) the switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", ["dict", "native"])
@pytest.mark.parametrize("recv", [False, True])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_equals_base(kind, seed, recv, how):
    t = _pair(kind, seed, dict(RPE, **(RECV if recv else {})))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S19_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S19_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", ["dict", "native"])
def test_count_only_equals_base(kind, seed):
    t = _pair(kind, seed, dict(RPE, **RECV))
    t.b.g.config.update(dict(ON, diffpc_receiver_count_only=True))
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S19_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_keys_not_in_default_config():
    for k in S19_SWITCHES + S19_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


def test_params_default_when_on():
    assert NF.Graph(dict(ON))._stab_diffpc_receiver() == (0.5, 64, 5, False)
    assert NF.Graph(dict(ON, diffpc_receiver_count_only=True))._stab_diffpc_receiver() == (0.5, 64, 5, True)
    assert NF.Graph()._stab_diffpc_receiver() is None
    assert NF.Graph(dict(RECV))._stab_diffpc_receiver() is None


def _errs(caplog):
    return [r for r in caplog.records if r.levelno == logging.ERROR and "diffpc_receiver_enabled" in r.getMessage()]


def test_needs_receiver(caplog):
    t = _pair("native", 0, dict(RPE))
    t.b.g.config.update(diffpc_receiver_enabled=True)
    with caplog.at_level(logging.ERROR):
        assert t.b.g._stab_diffpc_receiver() is None
        S.drive(t, random.Random(0), 10, sleep_every=True)
    assert len(_errs(caplog)) == 1, [r.getMessage() for r in _errs(caplog)]
    t.b.g.config.pop("diffpc_receiver_enabled")
    assert ckpt(t.a.g) == ckpt(t.b.g)


@pytest.mark.parametrize("bad", [dict(diffpc_receiver_eta=-0.1), dict(diffpc_receiver_eta=1.5),
                                 dict(diffpc_receiver_k=0), dict(diffpc_receiver_k="many"),
                                 dict(diffpc_receiver_window=0), dict(diffpc_receiver_window=6),
                                 dict(diffpc_receiver_eta=float("nan"))])
def test_bad_params_error_once(bad, caplog):
    g = NF.Graph(dict(ON, **bad))
    with caplog.at_level(logging.ERROR):
        assert g._stab_diffpc_receiver() is None
        assert g._stab_diffpc_receiver() is None
    assert len(_errs(caplog)) == 1, [r.getMessage() for r in _errs(caplog)]


# ---------------------------------------------------------------------------
# (2) the error on the arrival clock
# ---------------------------------------------------------------------------

def _dg(theta=0.8, fr=0.2, **cfg):
    """A dict graph with the corrected DiffPC on (count-only unless told otherwise) and one excitatory target T."""
    g = NF.Graph(dict(ON, diffpc_receiver_count_only=cfg.pop("count_only", True), **cfg))
    g.create_node(node_id="T")
    g.nodes["T"].threshold = theta
    g.nodes["T"].firing_rate_ema = fr
    g.create_node(node_id="z")              # a silent input, so T is never collected as an orphan
    g.create_synapse("z", "T", weight=0.01)
    return g


def _pend(g, target, current, dt=1):
    g._delay_buffer.setdefault(g.timestep + dt, []).append((target, current))


def _predict(g, target, expires_in=5, strength=0.5):
    p = NF.Prediction(source_node_id="s", target_node_id=target, strength=strength, confidence=0.5,
                      created_at=g.timestep, expires_at=g.timestep + expires_in, contributors=[],
                      drive=0.3, rho=0.3)
    g.active_predictions[p.prediction_id] = p
    return p


def _last(g):
    return g._diffpc_last


def test_surprise_external_input():
    g = _dg()
    g.stimulate("T", 2.0)
    r = g.step()
    assert "T" in r.fired_node_ids
    conf, surp = _last(g)
    assert conf == [] and surp == [("T", pytest.approx(0.8))]
    assert g._diffpc_stats["surprised"] == 1


def test_not_surprise_when_arrivals_fire_it():
    g = _dg()
    g.create_node(node_id="s0")
    g.create_synapse("s0", "T", weight=2.0)
    _pend(g, "T", 2.0)
    r = g.step()
    assert "T" in r.fired_node_ids
    assert _last(g) == ([], [])
    assert g._diffpc_stats["surprised"] == 0


def test_partial_arrivals_still_surprise():
    g = _dg()
    _pend(g, "T", 0.3)
    g.stimulate("T", 2.0)
    g.step()
    assert [nid for nid, _ in _last(g)[1]] == ["T"]


def test_he_injection_counts_as_explained():
    g = _dg()
    for m in ("m0", "m1"):
        g.create_node(node_id=m)
        g.nodes[m].threshold = 0.5
    g.create_hyperedge({"m0", "m1"}, activation_threshold=0.5, output_targets=["T"], output_weight=2.0)
    g.stimulate("m0", 2.0)
    g.stimulate("m1", 2.0)
    r1 = g.step()
    assert {"m0", "m1"} <= set(r1.fired_node_ids) and r1.fired_hyperedge_ids
    r2 = g.step()
    assert "T" in r2.fired_node_ids
    assert "T" not in [nid for nid, _ in _last(g)[1]]


def test_confirmed_not_surprised():
    g = _dg()
    _predict(g, "T")
    g.stimulate("T", 2.0)
    g.step()
    assert _last(g) == (["T"], [])
    assert g._diffpc_stats["confirmed"] == 1 and g._diffpc_stats["surprised"] == 0


def test_miss_counted():
    g = _dg()
    _predict(g, "T", expires_in=1)
    old = NF.Prediction(source_node_id="s", target_node_id="T", strength=0.5, confidence=0.5,
                        created_at=g.timestep, expires_at=g.timestep + 1)   # source-side: not DiffPC's
    g.active_predictions[old.prediction_id] = old
    for _ in range(3):
        g.step()
    assert g._diffpc_stats["misses"] == 1


def test_inhibitory_never_surprised():
    g = _dg()
    g.create_node(node_id="I", is_inhibitory=True)
    g.nodes["I"].threshold = 0.8
    g.stimulate("I", 2.0)
    _predict(g, "I")
    r = g.step()
    assert "I" in r.fired_node_ids
    assert _last(g) == ([], [])
    assert g._diffpc_stats["surprised"] == 0 and g._diffpc_stats["confirmed"] == 0


def _fire_T(g, predicted):
    if predicted:
        _predict(g, "T")
    g.stimulate("T", 2.0)
    r = g.step()
    assert "T" in r.fired_node_ids
    for _ in range(10):   # past refractory
        g.step()


def test_learned_from_surprise_counts():
    g = _dg()
    _fire_T(g, predicted=False)          # surprise
    _fire_T(g, predicted=True)           # next firing predicted -> learned
    st = g._diffpc_stats
    assert (st["surprised"], st["followups"], st["learned"]) == (1, 1, 1)
    _fire_T(g, predicted=False)          # surprise again
    _fire_T(g, predicted=False)          # next firing unpredicted (also a surprise) -> followup, not learned
    assert (st["surprised"], st["followups"], st["learned"]) == (3, 2, 1)


def test_followup_window_is_200_steps():
    g = _dg()
    _fire_T(g, predicted=False)
    for _ in range(200):
        g.step()
    _fire_T(g, predicted=True)
    assert g._diffpc_stats["followups"] == 0


@pytest.mark.parametrize("kind", ["dict", "native"])
def test_count_only_counts_and_old_body_runs(kind):
    t = _pair(kind, 0, dict(RPE, **RECV))
    t.b.g.config.update(dict(ON, diffpc_receiver_count_only=True))
    S.drive(t, random.Random(0), 25, sleep_every=True)
    st = t.b.g._diffpc_stats
    assert st["surprised"] > 0 and st["confirmed"] + st["misses"] > 0, st
    for k in S19_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@pytest.mark.parametrize("kind", ["dict", "native"])
def test_injection_recorder_reads_the_voltage_change(kind):
    g = restore_as(NF, kind, H2.built(0))
    g.config.update(ON)
    nid = next(iter(g.nodes.keys()))
    before = g._diffpc_voltages()
    g.nodes[nid].voltage += 0.5
    g._diffpc_record_injection(before)
    ex = g.nodes[nid].intrinsic_excitability
    assert g._diffpc_ring[g.timestep + 1] == {nid: pytest.approx(0.5 * g.config["decay_rate"] / ex)}


# ---------------------------------------------------------------------------
# (3) the |e| record, its one reset, the old body silent
# ---------------------------------------------------------------------------

def test_ema_abs_updates():
    g = _dg(count_only=False, diffpc_receiver_eta=0.0)
    g.nodes["T"].pred_error_ema = 0.7          # the old clock's value: reset at switch-on
    g.stimulate("T", 2.0)
    g.step()                                   # surprise, sigma 0.8
    e1 = g.nodes["T"].pred_error_ema
    assert e1 == pytest.approx(0.08)
    for _ in range(10):
        g.step()
    e1 = g.nodes["T"].pred_error_ema
    _predict(g, "T", expires_in=1, strength=0.5)
    for _ in range(3):
        g.step()                               # the miss, |e| = 0.5
    assert g._diffpc_stats["misses"] == 1
    assert g.nodes["T"].pred_error_ema == pytest.approx(0.9 * e1 + 0.05)


def test_confirmed_decays_ema():
    g = _dg(count_only=False, diffpc_receiver_eta=0.0)
    g.step()
    g.nodes["T"].pred_error_ema = 0.5
    _predict(g, "T")
    g.stimulate("T", 2.0)
    g.step()
    assert g.nodes["T"].pred_error_ema == pytest.approx(0.45)


def _layered(count_only):
    g = _dg(count_only=count_only, diffpc_receiver_eta=0.0)
    g.create_node(node_id="S")
    g.nodes["S"].threshold = 0.5
    g.create_synapse("S", "T", weight=0.1)
    g.nodes["S"].diffpc_layer, g.nodes["T"].diffpc_layer = 2, 1
    sid = next(iter(g._outgoing["S"]))
    g.stimulate("S", 2.0)
    g.step()
    return g, g.synapses.get(sid)


def test_old_body_silent_when_on():
    g, syn = _layered(count_only=True)
    assert g.nodes["S"].pred_weights            # control: count-only runs today's body
    g, syn = _layered(count_only=False)
    assert g.nodes["S"].pred_weights == {} and syn.eligibility_trace == 0.0


def test_step_result_reports_corrected_counts():
    g = _dg(count_only=False, diffpc_receiver_eta=0.0)
    g.stimulate("T", 2.0)
    r = g.step()
    assert r.diffpc_ternary_spikes == 1 and r.diffpc_mean_pred_error == pytest.approx(0.8)


def _reload(g):
    h = restore_as(NF, "dict", ckpt(g))
    h.config.update({k: v for k, v in g.config.items() if k.startswith(("diffpc_receiver", "prediction_receiver"))})
    return h


def test_reset_once_across_reload():
    g = _dg(count_only=False, diffpc_receiver_eta=0.0)
    g.nodes["z"].pred_error_ema = 0.4
    g.step()
    assert g.nodes["z"].pred_error_ema == 0.0
    g.nodes["z"].pred_error_ema = 0.4
    h = _reload(g)
    assert h.nodes["z"].pred_error_ema == pytest.approx(0.4)
    h.step()
    assert h.nodes["z"].pred_error_ema == pytest.approx(0.4)   # z is silent: no outcome, no second reset
    assert "_diffpc_stats" not in _reload(h).__dict__


def test_reset_flag_only_saved_when_set():
    g = _dg(count_only=True)
    g.step()
    assert "diffpc_receiver_reset_done" not in g._serialize_full()
    g.config["diffpc_receiver_count_only"] = False
    g.step()
    assert g._serialize_full()["diffpc_receiver_reset_done"] is True


def test_stats_not_in_checkpoint():
    g = _dg(count_only=False, diffpc_receiver_eta=0.0)
    g.stimulate("T", 2.0)
    g.step()
    base = NF.Graph(dict(RECV))
    base.create_node(node_id="T")
    extra = set(g._serialize_full()) - set(base._serialize_full())
    assert extra == {"diffpc_receiver_reset_done"}
