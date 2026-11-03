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
