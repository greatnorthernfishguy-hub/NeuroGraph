# ---- Changelog ----
# [2026-10-26] Claude (lane stability-s9, #1136; Josh 2026-10-26 "you have my 'go' when S8 is done") — S9: the reward
#   commits only the links of each prediction event
# What: (1) the S9 key absent / explicitly False == the S8 + #1146 NG branch (e49e33d's module) exactly on the dict,
#       NodeStore-only and native paths with the S7b arm's keys (the H2 drive workload incl. sweeps + sleeps; final
#       checkpoint BYTES); not in DEFAULT_CONFIG; set keys that cannot run log ERROR once and change nothing. (2) on a
#       native graph: a confirmed A->B commits only A->B; a failed A->B with C fired commits only A->B and A->C — an
#       unrelated D->C with a live trace is untouched (the base's broadcast moves it); inject_reward(post_scope=) without
#       a scope / without the S9 store is refused. (3) nothing new is saved.
# Why:  S8 dev-log §10/§11 (the surprise broadcast is the remaining waking climb); Josh's model of surprise vs confirmation.
# How:  oracle via `git show e49e33d:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers.
# -------------------
"""#1136 round S9: event-scoped reward commits."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S9_BASE_REV", "e49e33d")
CMOD = H2._load(BASE_REV, "nf_s8_commit_s9")
HAVE_S9 = getattr(NF.ng_tract.SynapseStore, "reward_round", 0) >= 1
needs_s9 = pytest.mark.skipif(not HAVE_S9, reason="ng_tract without the S9 reward pass")
S9_SWITCHES = ("reward_event_scoped_enabled",)
S9_PARAMS = ("reward_confirm_scaling",)
ARM = S7B.S7B_ARM
ckpt = S.ckpt
restore_as = S.restore_as


@needs_s9
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s8_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(ARM))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S9_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S9_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s9
def test_keys_not_in_default_config():
    for k in S9_SWITCHES + S9_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@needs_s9
@pytest.mark.parametrize("case", ["no_three_factor", "bad_scaling", "text_scaling"])
def test_misconfigured_s9_logs_error_once_and_changes_nothing(case, caplog):
    kind = "native"   # S9 needs only the native SYNAPSE store, which every graph kind here has
    extra = dict(reward_event_scoped_enabled=True)
    base = dict(ARM) if kind == "native" else {}
    if case == "no_three_factor":
        base["three_factor_enabled"] = False
    elif case == "bad_scaling":
        extra["reward_confirm_scaling"] = 1.5
    elif case == "text_scaling":
        extra["reward_confirm_scaling"] = "small"
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), 0)
    t.op("keys", lambda g: g.config.update(base))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and "reward_event_scoped_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the rule on a native graph
# ---------------------------------------------------------------------------

def _event_graph(m, on):
    g = m.Graph({"three_factor_enabled": True, "learning_rate": 0.1}, native_node_store=True, native_hyperedge_store=True)
    for n in "ABCD":
        g.create_node(node_id=n)
    syn = {k: g.create_synapse(k[0], k[1], weight=0.5).synapse_id for k in ("AB", "AC", "DC", "DB")}
    for k, sid in syn.items():
        g.synapses.get(sid).eligibility_trace = 0.4
    if on:
        g.config["reward_event_scoped_enabled"] = True
    return g, syn


def _w(g, syn):
    return {k: g.synapses.get(sid).weight for k, sid in syn.items()}


def _pred(m, a, b, conf=0.8):
    return m.Prediction(source_node_id=a, target_node_id=b, strength=0.5, confidence=conf)


@needs_s9
def test_confirmed_prediction_commits_only_its_own_link():
    g, syn = _event_graph(NF, True)
    w0 = _w(g, syn)
    g._on_prediction_confirmed(_pred(NF, "A", "B"), 0.1)
    w1 = _w(g, syn)
    assert w1["AB"] > w0["AB"]
    assert w1["AC"] == w0["AC"] and w1["DC"] == w0["DC"] and w1["DB"] == w0["DB"]
    # the base never commits on a confirmation (only a later broadcast does)
    b, bs = _event_graph(CMOD, False)
    b._on_prediction_confirmed(_pred(CMOD, "A", "B"), 0.1)
    assert _w(b, bs) == w0


@needs_s9
def test_surprise_commits_only_the_events_links():
    g, syn = _event_graph(NF, True)
    w0 = _w(g, syn)
    g._on_prediction_error(_pred(NF, "A", "B"), 0.1, {"C"})
    w1 = _w(g, syn)
    assert w1["AC"] != w0["AC"], "A -> what actually fired: committed"
    assert w1["AB"] != w0["AB"], "the wrong link's penalized trace: committed"
    assert w1["DC"] == w0["DC"] and w1["DB"] == w0["DB"], "unrelated links: untouched"
    # the base broadcasts to every trace
    b, bs = _event_graph(CMOD, False)
    b._on_prediction_error(_pred(CMOD, "A", "B"), 0.1, {"C"})
    wb = _w(b, bs)
    assert wb["DC"] != w0["DC"] and wb["DB"] != w0["DB"]


@needs_s9
def test_post_scope_is_refused_without_scope():
    g, _ = _event_graph(NF, True)
    with pytest.raises(ValueError):
        g.inject_reward(0.1, None, {"B"})


@needs_s9
def test_nothing_new_is_saved():
    g, _ = _event_graph(NF, True)
    g._on_prediction_error(_pred(NF, "A", "B"), 0.1, {"C"})
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    off, _ = _event_graph(NF, False)
    assert set(d) == set(msgpack.unpackb(ckpt(off), raw=False, strict_map_key=False))
