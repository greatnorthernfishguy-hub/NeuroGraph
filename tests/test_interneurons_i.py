# ---- Changelog ----
# [2026-10-20] Claude (lane interneurons-i, #1151 / #1136 round I; Josh 2026-10-20 "I'm all in on the inhibitory neurons,
#   as long as we don't do them in Python" + "It looks good as laid out") — the native interneuron layer: off == the S5
#   commit, S5 keys alone == S5 bitwise, never-surface, recruitment, divisive delivery, guard, iSTDP, checkpoint
# What: (1) every round-I key absent / explicitly False == the S5 commit (c4910e0's module) exactly on the dict,
#       NodeStore-only and native paths (every op of the H2 drive workload + a sleep; final checkpoint BYTES); the S5
#       candidate's keys alone (+ orphan round-I parameters) == S5 bitwise; misconfigured keys log ERROR once and
#       change nothing. (2) never surface: recruitment leaves graph.nodes / the E checkpoint keys unchanged, PV ids are
#       never in fired_node_ids, the layer holds E ids and numbers only. (3) recruitment is deterministic across hash
#       seeds and processes, and content-blind (stripping every node's metadata gives the same layer). (4) dynamics:
#       the layer fires and inhibits in step(), divisive is the default, the guard holds back failure samples, fixed
#       weights stay fixed (I1) and iSTDP moves them (I2); a pre_fire handler keeps it off. (5) checkpoint: save ->
#       load -> save byte-stable; the key is last and every other key is unchanged; a file with the layer and the keys
#       absent is inert (same steps as the stripped file) and carried verbatim; older readers step on it.
# Why:  ~/docs/superpowers/specs/2026-10-20-inhibitory-interneurons-design.md §4.1, §5.4-§5.6, §6 behaviour tests.
# How:  oracle loaded with `git show <rev>:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S5
#       helpers reused; PYTHONHASHSEED fixed by the runner (the determinism test spawns its own processes).
# -------------------
"""#1151 round I: the native PV interneuron layer."""
import hashlib
import json
import logging
import os
import random
import subprocess
import sys

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s5 as S5          # noqa: E402

NF = H2.NF
S5_REV = os.environ.get("NG_I_S5_REV", "c4910e0")   # the S5 commit (daa9503's neuro_foundation.py + its tests)
S5MOD = H2._load(S5_REV, "nf_s5_commit_i")
HAVE_I = (S5.HAVE_S5 and getattr(NF.ng_tract.NodeStore, "stab_round", 0) >= 6
          and hasattr(NF.ng_tract.NodeStore, "interneurons_recruit"))
needs_i = pytest.mark.skipif(not HAVE_I, reason="ng_tract without the round-I interneuron layer")

I_SWITCHES = ("interneurons_enabled", "interneuron_recruit_enabled", "interneuron_istdp_enabled")
I_PARAMS = {"interneuron_pool_max": 8, "interneuron_k_min": 2, "interneuron_w_min": 0.0, "interneuron_ratio_max": 0.5,
            "interneuron_w_ie": 0.7, "inh_g_max": 2.0, "interneuron_delay": 0, "interneuron_form": "offset"}
# small H2 graphs: let recruitment find structure (k_min 2, any weight)
SMALL = {"interneuron_k_min": 2, "interneuron_w_min": 0.0, "interneuron_ratio_max": 0.5}
I1 = dict(SMALL, interneurons_enabled=True, interneuron_recruit_enabled=True)
I2 = dict(I1, interneuron_istdp_enabled=True, interneuron_istdp_eta=0.01)
S5_FULL = S5.S5_FULL
ckpt = S.ckpt
restore_as = S.restore_as


def _top(b):
    """the checkpoint's top-level map as (key, raw value bytes) in file order"""
    u = msgpack.Unpacker(raw=False, strict_map_key=False, max_buffer_size=len(b) + 1)
    u.feed(b)
    out = []
    for _ in range(u.read_map_header()):
        k = u.unpack()
        a = u.tell()
        u.skip()
        out.append((k, b[a:u.tell()]))
    return out


def _grown(seed=0, cfg=I1, steps=0):
    g = restore_as(NF, "native", H2.built(seed))
    g.config.update(cfg)
    st = g._stability_params()
    rec = g._interneurons_recruit(st.irecruit)
    for _ in range(steps):
        g.step()
    return g, rec


# ---------------------------------------------------------------------------
# (1) identities
# ---------------------------------------------------------------------------

@needs_i
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s5_commit_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(S5MOD, kind, start), restore_as(NF, kind, start), seed)
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in I_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in I_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    assert not hasattr(t.b.g, "_stab")
    if kind != "dict":
        assert not t.b.g.nodes.interneurons_exists()


@needs_i
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("keys", ["s5_full", "s5_orphan_i_params"])
def test_s5_keys_alone_equal_the_s5_commit_bitwise(seed, keys):
    start = H2.built(seed)
    t = S.Pair(restore_as(S5MOD, "native", start), restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(S5_FULL))
    if keys == "s5_orphan_i_params":
        t.b.g.config.update(I_PARAMS)
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in I_PARAMS:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    assert t.b.g._stab.ilayer is None and t.b.g._stab.irecruit is None


@needs_i
@pytest.mark.parametrize("case", ["dict_graph", "bad_form", "bad_delay", "bad_ratio", "bad_gmax", "pre_fire"])
def test_misconfigured_i_keys_log_error_once_and_change_nothing(case, caplog):
    kind = "dict" if case == "dict_graph" else "native"
    extra = dict(I2)
    extra.update({"dict_graph": {}, "bad_form": {"interneuron_form": "subtractive", "interneuron_recruit_enabled": False},
                  "bad_delay": {"interneuron_delay": 2, "interneuron_recruit_enabled": False},
                  "bad_ratio": {"interneuron_ratio_max": 1.5, "interneurons_enabled": False},
                  "bad_gmax": {"inh_g_max": 0.0, "interneuron_recruit_enabled": False},
                  "pre_fire": {}}[case])
    start = H2.built(0)
    t = S.Pair(restore_as(S5MOD, kind, start), restore_as(NF, kind, start), 0)
    if case == "pre_fire":
        for g in (t.a.g, t.b.g):
            g.register_event_handler("pre_fire", lambda *a, **k: 0.0)
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage() and r.levelno == logging.ERROR]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    if kind != "dict":
        assert not t.b.g.nodes.interneurons_exists()


@needs_i
def test_i_keys_not_in_default_config_and_default_config_is_the_s5_tips():
    for k in I_SWITCHES + tuple(I_PARAMS) + ("interneuron_istdp_eta", "interneuron_istdp_tau", "interneuron_istdp_wmax",
                                             "interneuron_theta", "interneuron_coincidence", "interneuron_refractory",
                                             "interneuron_decay", "inh_g_decay", "inh_g_eps", "interneuron_cover",
                                             "interneuron_retire_window"):
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(S5MOD.DEFAULT_CONFIG.items())


# ---------------------------------------------------------------------------
# (2) never surface + recruitment never touches E structure
# ---------------------------------------------------------------------------

@needs_i
@pytest.mark.parametrize("seed", [0, 1])
def test_recruitment_leaves_every_e_key_and_graph_nodes_unchanged(seed):
    g = restore_as(NF, "native", H2.built(seed))
    g.config.update(I1)
    before = _top(ckpt(g))
    ids_before = list(g.nodes.keys())
    rec = g._interneurons_recruit(g._stability_params().irecruit)
    assert rec["cells"] > 0 and rec["recruited"] == rec["cells"]
    assert rec["cells"] <= 0.5 * rec["eligible_e"]
    after = _top(ckpt(g))
    assert [k for k, _ in after] == [k for k, _ in before] + ["interneurons"]   # the new key, last
    assert after[:-1] == before                                              # every other key byte-identical
    assert list(g.nodes.keys()) == ids_before and len(g.nodes) == len(ids_before)
    _, _, _, pools = g.nodes.interneuron_arrays(pools=True)
    assert all(m in g.nodes for p in pools for m in p)                       # E ids only


@needs_i
def test_pv_cells_never_appear_in_fired_ids_or_graph_nodes():
    g, _ = _grown(0, I2)
    n0, ids0 = len(g.nodes), set(g.nodes.keys())
    fired_pv = 0
    for k in range(60):
        if k % 6 == 0:
            for nid in list(ids0)[:10]:
                g.stimulate(nid, 2.0)
        r = g.step()
        assert set(r.fired_node_ids) <= set(g.nodes.keys())
        fired_pv += g.nodes.interneuron_census(0.01)["fires_last"]
    assert fired_pv > 0, "the PV cells fired, and none of them surfaced"
    assert len(g.nodes) <= n0 and set(g.nodes.keys()) <= ids0


@needs_i
def test_recruitment_is_content_blind():
    a, _ = _grown(0)
    g = restore_as(NF, "native", H2.built(0))
    for nid in list(g.nodes.keys()):
        g.nodes[nid].metadata = {}                                           # delete every word of content
    g.config.update(I1)
    g._interneurons_recruit(g._stability_params().irecruit)
    assert g.nodes.interneurons_to_msgpack() == a.nodes.interneurons_to_msgpack()


_DET = r"""
import sys, hashlib
sys.path.insert(0, {wt!r}); import os; os.chdir({wt!r})
from tests import test_hyperedge_store_h2 as H2
from tests import test_stability_s as S
g = S.restore_as(H2.NF, "native", H2.built(0))
g.config.update({cfg!r})
g._interneurons_recruit(g._stability_params().irecruit)
print(hashlib.sha256(g.nodes.interneurons_to_msgpack()).hexdigest())
"""


@needs_i
def test_recruitment_is_identical_across_hash_seeds_and_processes():
    wt = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = set()
    for hs in ("0", "1", "2", "3", "0"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", _DET.format(wt=wt, cfg=I1)], env=env, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.add(r.stdout.strip().splitlines()[-1])
    assert len(outs) == 1, outs


# ---------------------------------------------------------------------------
# (3) dynamics
# ---------------------------------------------------------------------------

def _drive(g, steps, kick=2.0):
    ids = list(g.nodes.keys())
    for k in range(steps):
        if k % 6 == 0:
            for nid in ids[:12]:
                g.stimulate(nid, kick)
        g.step()


@needs_i
def test_the_layer_inhibits_in_step_and_changes_e_firing_divisively_by_default():
    g0 = restore_as(NF, "native", H2.built(0))
    g0.config.update({k: v for k, v in I1.items() if k != "interneurons_enabled"})   # recruit, but no dynamics
    g1, _ = _grown(0, I1)
    g0._interneurons_recruit(g0._stability_params().irecruit)
    f0 = f1 = 0
    ids = list(g0.nodes.keys())
    inhibited = 0
    for k in range(80):
        if k % 6 == 0:
            for nid in ids[:12]:
                g0.stimulate(nid, 2.0)
                g1.stimulate(nid, 2.0)
        f0 += len(g0.step().fired_node_ids)
        f1 += len(g1.step().fired_node_ids)
        inhibited = max(inhibited, g1.nodes.interneuron_census(0.01)["inhibited"])
    assert g1._stab.ilayer[6] == 0.0                     # divisive is the default form
    assert inhibited > 0 and f1 < f0, (f0, f1)
    c0 = g0.nodes.interneuron_census(0.01)
    assert c0["steps"] == 0 and c0["inhibited"] == 0     # keys absent for the dynamics: the layer is inert


@needs_i
def test_fixed_weights_stay_fixed_in_i1_and_istdp_moves_them_in_i2():
    g1, _ = _grown(0, I1)
    g2, _ = _grown(0, I2)
    _drive(g1, 120)
    _drive(g2, 120)
    w1 = g1.nodes.interneuron_arrays()[1]
    w2 = g2.nodes.interneuron_arrays()[1]
    assert set(w1) == {0.5}                              # I1: interneuron_w_ie default, untouched
    assert set(w2) != {0.5} and min(w2) >= 0.0 and max(w2) <= 3.0   # I2: learned, clipped to [0, w_max] (this small
    # graph is driven far over its set point, so its inhibition climbs; convergence both ways: the Rust unit test)


@needs_i
def test_the_guard_counts_held_back_failures_with_input_evidence_on():
    cfg = dict(I1, node_homeo_evidence_enabled=True, node_homeo_input_evidence_enabled=True)
    g, _ = _grown(0, cfg)
    _drive(g, 80)
    assert g.nodes.interneuron_census(0.01)["guard_hits"] > 0


# ---------------------------------------------------------------------------
# (4) checkpoint
# ---------------------------------------------------------------------------

@needs_i
def test_save_load_save_is_byte_stable_and_the_learned_weights_survive():
    g, _ = _grown(0, I2)
    _drive(g, 60)
    b1 = ckpt(g)
    assert _top(b1)[-1][0] == "interneurons"
    w = g.nodes.interneuron_arrays()[1]
    g2 = restore_as(NF, "native", b1)
    assert g2.nodes.interneurons_exists() and g2.nodes.interneuron_arrays()[1] == w
    assert _top(ckpt(g2))[-1] == _top(b1)[-1]           # the layer re-saves byte-identically
    g3 = restore_as(NF, "native", ckpt(g2))
    assert g3.nodes.interneurons_to_msgpack() == g.nodes.interneurons_to_msgpack()


@needs_i
def test_a_file_with_the_layer_and_the_keys_absent_is_inert_and_carried_verbatim():
    g, _ = _grown(0, I1)
    for k in I1:
        g.config.pop(k)
    b = ckpt(g)
    layer = _top(b)[-1]
    stripped = msgpack.packb({k: msgpack.unpackb(v, raw=False, strict_map_key=False) for k, v in _top(b)[:-1]},
                             use_bin_type=True)
    # carried verbatim: load -> plain steps (no E node removed) -> save gives the identical layer bytes
    gc = restore_as(NF, "native", b)
    n0 = len(gc.nodes)
    for _ in range(10):
        gc.step()
    assert len(gc.nodes) == n0 and _top(ckpt(gc))[-1] == layer
    # inert: the full workload (steps, sweeps, rewards, node removal, sleep) runs exactly as on the stripped file;
    # only the layer key differs (dead edges of removed nodes are dropped on save)
    ga = restore_as(NF, "native", b)
    gb = restore_as(NF, "native", stripped)
    assert ga.nodes.interneurons_exists() and not gb.nodes.interneurons_exists()
    t = S.Pair(ga, gb, 0)
    S.drive(t, random.Random(0), 12, sleep_every=True)
    assert [x for x in _top(ckpt(ga)) if x[0] != "interneurons"] == _top(ckpt(gb))
    assert ga.nodes.interneuron_census(0.01)["steps"] == 0


@needs_i
def test_older_readers_step_on_a_file_with_the_layer_and_a_resave_drops_it():
    g, _ = _grown(0, I2)
    _drive(g, 30)
    b = ckpt(g)
    for M, kind in ((S5MOD, "native"), (S5MOD, "dict"), (S.TIP, "dict"), (NF, "dict")):
        h = restore_as(M, kind, b)
        for _ in range(3):
            h.step()
        keys = [k for k, _ in _top(ckpt(h))]
        assert "interneurons" not in keys                # the documented lossy downgrade
