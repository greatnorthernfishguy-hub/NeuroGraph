# ---- Changelog ----
# [2026-10-26] Claude (lane stability-s8, #1146; Josh 2026-10-26 "Works for me") — the graph's own HomeostaticRule follows
#   graph.config live
# What: (1) a host's graph.config change of target_firing_rate / degree_sensitivity / scaling_interval after restore()
#       reaches the rule at its next apply (the S8 branch 32b78b7 keeps the checkpoint's values: the bug); degree
#       targets are refreshed at once; (2) unchanged config == the S8 branch exactly (dict / NodeStore / native x 3
#       seeds, the S7b arm's keys, checkpoint bytes); (3) a hand-built rule keeps its own values.
# Why:  #1146 (punchlist/open/neurograph.md); dev-log 2026-10-25_stability-s8-scaling-total-bound §9.
# How:  oracle via `git show 32b78b7:neuro_foundation.py`; the S / S7b helpers.
# -------------------
"""#1146: the homeostat takes the host's config on the same boot."""
import os
import random

import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_1146_BASE_REV", "32b78b7")
CMOD = H2._load(BASE_REV, "nf_s8_commit_1146")


def _rule(g, m=NF):
    return next(r for r in g._plasticity_rules if isinstance(r, m.HomeostaticRule))


def _stepped(m, kind, data, cfg, steps=6):
    g = S.restore_as(m, kind, data)
    g.config.update(cfg)
    ids = list(g.nodes)[:5]
    for _ in range(steps):
        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()
    return g


@pytest.mark.parametrize("kind", ["dict", "native"])
def test_host_change_after_restore_reaches_the_rule_at_once(kind):
    data = S.ckpt(S.restore_as(NF, kind, H2.built(0)))
    cfg = dict(target_firing_rate=0.02, degree_sensitivity=0.7, scaling_interval=7)
    new = _stepped(NF, kind, data, cfg)
    r = _rule(new)
    assert (r.target_firing_rate, r.degree_sensitivity, r.scaling_interval) == (0.02, 0.7, 7)
    dt = list(r._degree_targets.values())
    assert dt and max(dt) <= 0.02 * 1.7 + 1e-12 and min(dt) >= 0.02 * 0.3 - 1e-12
    old = _stepped(CMOD, kind, data, cfg)
    ro = _rule(old, CMOD)
    assert ro.target_firing_rate != 0.02, "the base keeps the checkpoint's value (the bug)"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
def test_unchanged_config_is_the_s8_branch_exactly(kind, seed):
    start = H2.built(seed)
    t = S.Pair(S.restore_as(CMOD, kind, start), S.restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(S7B.S7B_ARM))
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    assert S.ckpt(t.a.g) == S.ckpt(t.b.g)


def test_a_hand_built_rule_keeps_its_own_values():
    g = NF.Graph({"target_firing_rate": 0.05, "scaling_interval": 100})
    rule = NF.HomeostaticRule(target_firing_rate=0.03, scaling_interval=2, degree_sensitivity=0.4)
    for i in range(4):
        g.create_node(node_id=f"n{i}")
    rule.apply(g, ["n0"], 1)
    assert (rule.target_firing_rate, rule.scaling_interval) == (0.03, 2)
    assert _rule(g)._follows_config is True and rule._follows_config is False
