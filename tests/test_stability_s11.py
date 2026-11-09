# ---- Changelog ----
# [2026-10-27] Claude (lane stability-s11, #1136; Josh 2026-10-27 "Go ahead and try the run overnight") — S11: C2 gets its
#   downward half (C2-down), evidence-gated
# What: (1) the S11 key absent / explicitly False == the S10 NG branch (82785b9's module) exactly on the dict, NodeStore-only
#       and native paths with the S7b arm's keys (final checkpoint BYTES); not in DEFAULT_CONFIG; set keys that cannot run
#       (no C2, bad rate / evidence / floor) log ERROR once and change nothing. (2) the rule: an under-active learnable
#       assembly that is nearly driven gets more excitable and comes to fire; one without input, or a non-learnable one,
#       is untouched; never below the floor; an over-active one still climbs (C2); nothing new is saved.
# Why:  dev-log 2026-10-27_stability-s10-rpe-reward §4.3.
# How:  oracle via `git show 82785b9:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers.
# -------------------
"""#1136 round S11: C2-down (an under-active, nearly-driven assembly becomes more excitable)."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S11_BASE_REV", "82785b9")
CMOD = H2._load(BASE_REV, "nf_s10_commit_s11")
S11_SWITCHES = ("he_threshold_homeo_down_enabled",)
S11_PARAMS = ("he_threshold_homeo_down_rate", "he_threshold_homeo_down_evidence", "he_threshold_homeo_down_floor")
ARM = S7B.S7B_ARM
ckpt = S.ckpt
restore_as = S.restore_as
HAVE_S11 = getattr(NF.ng_tract.HyperedgeStore, "c2_down_round", 0) >= 1
needs_s11 = pytest.mark.skipif(not HAVE_S11, reason="ng_tract without the S11 C2-down kernel")
C2 = {"he_threshold_homeo_enabled": True, "he_rate_target": 0.05, "he_threshold_homeo_rate": 0.01}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s10_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(ARM))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S11_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S11_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_keys_not_in_default_config():
    for k in S11_SWITCHES + S11_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@needs_s11
@pytest.mark.parametrize("case", ["no_c2", "zero_rate", "text_rate", "evidence_one", "floor_one"])
def test_misconfigured_s11_logs_error_once_and_changes_nothing(case, caplog):
    base = dict(ARM)
    extra = dict(he_threshold_homeo_down_enabled=True)
    if case == "no_c2":
        base["he_threshold_homeo_enabled"] = False
    elif case == "zero_rate":
        extra["he_threshold_homeo_down_rate"] = 0
    elif case == "text_rate":
        extra["he_threshold_homeo_down_rate"] = "fast"
    elif case == "evidence_one":
        extra["he_threshold_homeo_down_evidence"] = 1.0
    elif case == "floor_one":
        extra["he_threshold_homeo_down_floor"] = 1.0
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(base))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR
            and "he_threshold_homeo_down_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the rule
# ---------------------------------------------------------------------------

def _near(down, thr=0.6, members=4, driven=2, steps=60, **kw):
    """An assembly whose input reaches `driven` of `members` every step (just short of its threshold)."""
    cfg = dict(C2)
    if down:
        cfg.update({"he_threshold_homeo_down_enabled": True, "he_threshold_homeo_down_rate": 0.01}, **kw)
    g, x = S.tiny(cfg, n=members + 2)
    h = g.create_hyperedge(set(x[:members]), activation_threshold=thr).hyperedge_id
    g._plasticity_rules = []
    thrs, fires = [], 0
    for _ in range(steps):
        for i in range(driven):
            g.stimulate(x[i], 2.0)
        g.stimulate(x[-1], 2.0)        # (something fires every step)
        fires += h in g.step().fired_hyperedge_ids
        thrs.append(g.hyperedges[h].activation_threshold)
    return g, h, thrs, fires


@needs_s11
def test_a_nearly_driven_under_active_assembly_comes_to_fire():
    g0, h0, thrs0, fires0 = _near(False)
    g, h, thrs, fires = _near(True)
    assert thrs0[-1] == 0.6 and fires0 == 0, "S10 / S2: nothing lowers it; it never fires on half its input"
    assert thrs[0] == pytest.approx(0.59) and min(thrs) < 0.6
    assert fires > 0, "C2-down: it becomes excitable enough to complete on the input it keeps getting"


@needs_s11
def test_an_assembly_without_input_is_untouched():
    g, h, thrs, fires = _near(True, driven=0)
    assert thrs[-1] == 0.6 and fires == 0
    g, h, thrs, fires = _near(True, driven=1, thr=0.6)   # 0.25 < 0.5 x 0.6: below the evidence fraction
    assert thrs[-1] == 0.6


@needs_s11
def test_never_below_the_floor():
    g, h, thrs, fires = _near(True, thr=0.6, driven=2, steps=200, he_threshold_homeo_down_floor=0.55,
                              he_threshold_homeo_down_evidence=0.1)
    assert min(thrs) >= 0.55 and thrs[-1] == pytest.approx(0.55)


@needs_s11
def test_an_over_active_assembly_still_climbs():
    g, x = S.tiny(dict(C2, he_threshold_homeo_down_enabled=True, he_threshold_homeo_down_rate=0.01), n=8)
    hot = g.create_hyperedge({x[0], x[1]}, activation_threshold=0.3).hyperedge_id
    g.hyperedges[hot].recent_activation_ema = 0.3
    g._plasticity_rules = []
    for _ in range(100):
        g.stimulate(x[0], 2.0)
        g.stimulate(x[1], 2.0)
        g.step()
    assert g.hyperedges[hot].activation_threshold > 0.5


@needs_s11
def test_nothing_new_is_saved():
    g, h, thrs, fires = _near(True)
    off, h0, _, _ = _near(False)
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    d0 = msgpack.unpackb(ckpt(off), raw=False, strict_map_key=False)
    assert set(d) == set(d0)
    assert set(d["hyperedges"][h]) <= set(d0["hyperedges"][h0]) | {"c2_threshold_base"}
    back = restore_as(NF, "native", ckpt(g))
    assert back.hyperedges[h].activation_threshold == g.hyperedges[h].activation_threshold
