# ---- Changelog ----
# [2026-10-25] Claude (lane stability-s8, #1136 / #1141; Josh 2026-10-25 "Yessir" to the S8 draft + "a nodes totals should
#   be saved") — S8: a bound on S's TOTAL up-scaling per node
# What: (1) the S8 key absent / explicitly False == the S7b NG branch (00f388a's module) exactly on the dict, NodeStore-only
#       and native paths with the whole S7b arm on (the H2 drive workload incl. sweeps + sleeps; final checkpoint BYTES);
#       not in DEFAULT_CONFIG; set keys that cannot run log ERROR once and change nothing. (2) with the key the
#       stability params carry B, and on a real graph no node's total passes ln B while up-factors are granted, trimmed
#       and refused. (3) the totals are saved (syn_scale_up_totals) and restored; the S7b branch reads the file; a
#       sleep returns room by ln(1 - d0).
# Why:  hyperedge-store design §2.3 (S8 row); S8 weight ledger 2026-10-25.
# How:  oracle via `git show 00f388a:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S6 / S7b helpers.
# -------------------
"""#1136 round S8: the bound on synaptic scaling's total up-scaling per node."""
import logging
import math
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s6 as S6          # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
S7B_REV = os.environ.get("NG_S8_BASE_REV", "00f388a")
CMOD = H2._load(S7B_REV, "nf_s7b_commit_s8")
HAVE_S8 = getattr(NF.ng_tract.NodeStore, "stab_round", 0) >= 9 and hasattr(NF.ng_tract.NodeStore, "s8_totals")
needs_s8 = pytest.mark.skipif(not HAVE_S8, reason="ng_tract without the S8 kernel")
S8_SWITCHES = ("syn_scale_up_bound_enabled",)
S8_PARAMS = ("syn_scale_up_bound",)
S7B_ARM = S7B.S7B_ARM
S8_ARM = dict(S7B_ARM, syn_scale_up_bound_enabled=True)
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) identities
# ---------------------------------------------------------------------------

@needs_s8
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s7b_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(S7B_ARM))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S8_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S8_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s8
def test_keys_not_in_default_config():
    for k in S8_SWITCHES + S8_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@needs_s8
@pytest.mark.parametrize("case", ["no_s", "bound_one", "bound_under_one", "bound_nan", "bound_inf", "bound_text"])
def test_misconfigured_bound_logs_error_once_and_changes_nothing(case, caplog):
    cfg, extra = S7B_ARM, dict(syn_scale_up_bound_enabled=True)
    if case == "no_s":
        cfg = {k: v for k, v in S7B_ARM.items() if k not in ("syn_scale_evidence_enabled", "syn_scale_settle_enabled",
                                                             "syn_scale_quiet_hold_enabled")}
    else:
        extra.update({"bound_one": dict(syn_scale_up_bound=1.0), "bound_under_one": dict(syn_scale_up_bound=0.5),
                      "bound_nan": dict(syn_scale_up_bound=float("nan")),
                      "bound_inf": dict(syn_scale_up_bound=float("inf")),
                      "bound_text": dict(syn_scale_up_bound="two")}[case])
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(cfg))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage() and r.levelno == logging.ERROR
            and "syn_scale_up_bound_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s8
def test_bound_on_a_dict_backed_graph_logs_error_and_stays_off(caplog):
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "dict", start), restore_as(NF, "dict", start), 0)
    t.b.g.config.update(S8_ARM)
    t.a.g.config.update(S7B_ARM)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 10)
    assert any("S8-round" in r.getMessage() for r in caplog.records)
    t.b.g.config.pop("syn_scale_up_bound_enabled")
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the wiring and the bound on a real graph
# ---------------------------------------------------------------------------

def _graph(seed, cfg):
    return S6._graph(seed, cfg)


@needs_s8
def test_stability_params_carry_the_bound():
    assert _graph(0, S8_ARM)._stability_params().up_bound == 2.0
    assert _graph(0, dict(S8_ARM, syn_scale_up_bound=1.5))._stability_params().up_bound == 1.5
    assert _graph(0, S7B_ARM)._stability_params().up_bound is None


def _drive_weak(g, steps, frac=1.0):
    # every node gets a weak input that arrives but fails to fire it: driven and failing (S's up-scaling case)
    ids = list(g.nodes)[: max(1, int(len(g.nodes) * frac))]
    for _ in range(steps):
        for nid in ids:
            g.stimulate(nid, 0.05)
        g.step()


@needs_s8
def test_no_node_passes_the_bound_on_a_real_graph_and_down_scaling_returns_room():
    cfg = {k: v for k, v in S8_ARM.items() if not k.startswith("interneuron")}
    cfg.update(spontaneous_activity_enabled=False, syn_scale_up_bound=1.05)
    g = _graph(7, cfg)
    lb = math.log(1.05)
    _drive_weak(g, 300)
    st = g.nodes.s8_state(1.05)
    # the bound bites: granted, trimmed to the room, then refused; nobody past it
    assert st["up_passed"] > 0 and st["up_trimmed"] > 0 and st["up_refused"] > 0, st
    assert st["rows_at_bound"] > 0 and st["g_max"] <= lb + 1e-9, st
    at = st["rows_nonzero"]
    # the same weak drive later fires these nodes above target: down-scaling (never blocked) returns their room
    _drive_weak(g, 1200)
    st2 = g.nodes.s8_state(1.05)
    assert st2["g_max"] <= lb + 1e-9, st2
    assert st2["rows_nonzero"] < at, (st, st2)


# ---------------------------------------------------------------------------
# (3) the totals are saved; sleep returns room
# ---------------------------------------------------------------------------

@needs_s8
def test_totals_are_saved_restored_and_the_s7b_branch_reads_the_file():
    cfg = {k: v for k, v in S8_ARM.items() if not k.startswith("interneuron")}
    cfg.update(spontaneous_activity_enabled=False, syn_scale_up_bound=1.05)
    g = _graph(7, cfg)
    _drive_weak(g, 600)
    tot = dict(g.nodes.s8_totals())
    assert tot, "some totals"
    data = ckpt(g)
    d = msgpack.unpackb(data, raw=False, strict_map_key=False)
    assert d["syn_scale_up_totals"] == pytest.approx(tot)
    assert list(d)[-1] == "interneurons" or "interneurons" not in d
    h = restore_as(NF, "native", data)
    h.config.update(cfg)
    assert dict(h.nodes.s8_totals()) == pytest.approx(tot)
    # the S7b branch reads the file (it ignores the key)
    o = restore_as(CMOD, "native", data)
    assert len(o.nodes) == len(g.nodes)
    o.step()
    # off: the key is not written
    g.config["syn_scale_up_bound_enabled"] = False
    assert "syn_scale_up_totals" not in msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)


@needs_s8
def test_sleep_returns_room_by_ln_one_minus_d0():
    cfg = {k: v for k, v in S8_ARM.items() if not k.startswith("interneuron")}
    cfg.update(spontaneous_activity_enabled=False, syn_scale_up_bound=1.05)
    g = _graph(7, cfg)
    _drive_weak(g, 600)
    before = dict(g.nodes.s8_totals())
    g._s8_sleep_shrink(0.01)
    after = dict(g.nodes.s8_totals())
    for nid, v in before.items():
        assert after.get(nid, 0.0) == pytest.approx(max(0.0, v + math.log(0.99)))
    # off: nothing moves
    g.config["syn_scale_up_bound_enabled"] = False
    g._s8_sleep_shrink(0.5)
    assert dict(g.nodes.s8_totals()) == after
