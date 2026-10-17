# ---- Changelog ----
# [2026-10-16] Claude Code (Opus 5.5) — #1131 condition F (evoked minus spontaneous) tests
# What: a small synthetic graph with a self-sustaining loop (fires with no input) and a
#       cue -> evoked chain. Proves: switch ON -> the loop loses to what the cue evokes;
#       switch OFF -> identical to the base commit's recall (4e2ed30) and no extra sweep;
#       no graph mutation (full serialized state + transient voltages/refractory equal
#       before/after); the spontaneous sweep runs at the same depth as the harvest.
# Why:  dev-log 2026-10-16_cc-recall-reverberating-loop-1131; CC_RECALL_EVOKED must be a
#       pure read and byte-identical when off.
# How:  the REAL NeuroGraphMemory._harvest_associations bound to a stub ng (stub VDB seeds
#       the cue node; no embedding model), the real Graph.prime_and_propagate, GSG
#       re-score stubbed to identity (it needs the ONNX embedder; it is not under test).
# -------------------
import importlib.util
import math
import os
import subprocess
import sys
import types
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

import cc_ng_organism as org
import neuro_foundation as nf
from openclaw_hook import NeuroGraphMemory

LOOP_TEXT = "RULING packet one-three-eight five a: a self-sustaining chief tree"
LOOP2_TEXT = "RULING packet one-three-eight sibling tree that keeps it going"
CUE_TEXT = "how does the redis cache behave on a cold first boot"
EVOKED_TEXT = "the deploy pipeline breaks whenever the redis cache is cold"


class _StubVDB:
    """Seeds exactly the cue node (what a cosine search for the query would do)."""

    def __init__(self, contents):
        self._contents = contents

    def search(self, query_vec, k=10, threshold=0.0):
        return [("cue", 1.0)]

    def get(self, nid):
        text = self._contents.get(nid)
        return None if text is None else {"content": text, "metadata": {}}


def _build_ng():
    g = nf.Graph()
    contents = {"loop": LOOP_TEXT, "loop2": LOOP2_TEXT, "cue": CUE_TEXT, "evoked": EVOKED_TEXT}
    for nid in contents:
        g.create_node(node_id=nid)
    # The reverberating pair: strong mutual excitation, and 'loop' already above threshold,
    # so it fires at step 0 with NO stimulus and keeps 'loop2' going.
    g.create_synapse("loop", "loop2", weight=3.0)
    g.create_synapse("loop2", "loop", weight=3.0)
    g.nodes["loop"].voltage = 3.0
    g.nodes["loop"].threshold = 0.5
    g.nodes["loop2"].threshold = 0.5
    # The cue's own chain: cue fires on the prime, drives 'evoked' one step later.
    g.create_synapse("cue", "evoked", weight=1.0)
    g.nodes["cue"].threshold = 0.3
    g.nodes["evoked"].threshold = 0.3
    ng = SimpleNamespace(graph=g, vector_db=_StubVDB(contents),
                         ingestor=SimpleNamespace(embedder=SimpleNamespace(
                             embed_text=lambda text: [1.0, 0.0])))
    ng._harvest_associations = types.MethodType(NeuroGraphMemory._harvest_associations, ng)
    return ng


@pytest.fixture
def switch(monkeypatch):
    monkeypatch.setattr(org, "cc_gsg_rescore", lambda surfaced, *a, **kw: surfaced)

    def _set(on, oversample=6):
        monkeypatch.setattr(org, "_CC_RECALL_EVOKED", on)
        monkeypatch.setattr(org, "_CC_RECALL_EVOKED_OVERSAMPLE", oversample)
        monkeypatch.setattr(org, "_CC_RECALL_SELECTIVITY", False)
        monkeypatch.setattr(org, "_CC_RECALL_PROP_STEPS", 0)
    return _set


def _ids(results):
    return [r["node_id"] for r in results]


def test_off_the_loop_wins_on_the_evoked_node_wins(switch):
    switch(False)
    off = org.cc_pattern_completion_recall(_build_ng(), CUE_TEXT, k=1)
    assert _ids(off) == ["loop"], "precondition: today the self-sustaining loop takes the top slot"

    switch(True)
    on = org.cc_pattern_completion_recall(_build_ng(), CUE_TEXT, k=4)
    assert on, "the cue evoked something"
    assert "loop" not in _ids(on) and "loop2" not in _ids(on)
    assert set(_ids(on)) <= {"cue", "evoked"}
    assert "evoked" in _ids(on)
    assert all(r["score"] > 0 for r in on)
    assert [r["score"] for r in on] == sorted((r["score"] for r in on), reverse=True)


def test_net_score_is_evoked_minus_spontaneous(switch):
    """A node the cue drives ABOVE its spontaneous level keeps only the difference."""
    switch(True)
    ng = _build_ng()
    ng.graph.create_synapse("cue", "loop", weight=2.0)  # the cue also feeds the loop
    ng.graph.nodes["loop"].voltage = 0.0                # ... which now is quiet at rest,
    spont = org._cc_spontaneous_activity(ng.graph, 3)
    assert "loop" not in spont
    ng.graph.nodes["loop"].voltage = 3.0                # and with ongoing activity again:
    spont = org._cc_spontaneous_activity(ng.graph, 3)
    assert spont["loop"] > 0
    surfaced = [{"node_id": "loop", "strength": spont["loop"] + 0.25, "content": LOOP_TEXT},
                {"node_id": "evoked", "strength": 0.9, "content": EVOKED_TEXT},
                {"node_id": "loop2", "strength": spont["loop2"], "content": LOOP2_TEXT}]
    out = org._cc_evoked_over_spontaneous(ng.graph, surfaced, 3)
    assert _ids(out) == ["evoked", "loop"]
    assert out[1]["strength"] == pytest.approx(0.25)
    assert out[0]["strength"] == pytest.approx(0.9)       # no spontaneous activity -> unchanged


def _transient(g):
    return ({nid: (n.voltage, n.refractory_remaining, n.last_spike_time, n.firing_rate_ema,
                   n.threshold) for nid, n in g.nodes.items()},
            {hid: he.refractory_remaining for hid, he in g.hyperedges.items()},
            g.timestep)


def test_switch_on_does_not_mutate_the_graph(switch):
    switch(True)
    ng = _build_ng()
    before_full = ng.graph._serialize_full()
    before_t = _transient(ng.graph)
    org.cc_pattern_completion_recall(ng, CUE_TEXT, k=4)
    assert _transient(ng.graph) == before_t
    assert ng.graph._serialize_full() == before_full


def test_spontaneous_sweep_is_one_read_mode_call_at_the_harvest_depth(switch, monkeypatch):
    """Exactly one extra prime_and_propagate, write_mode False, same steps as the harvest --
    including the #255 novelty scaling and the CC_RECALL_PROP_STEPS override."""
    for prop_steps in (0, 2):
        for novelty in (0.0, 0.5, 1.0):
            switch(True)
            monkeypatch.setattr(org, "_CC_RECALL_PROP_STEPS", prop_steps)
            monkeypatch.setattr(org, "cc_novelty", lambda state, graph, _n=novelty: _n)
            ng = _build_ng()
            calls = []
            real = ng.graph.prime_and_propagate

            def spy(node_ids, currents, steps=3, write_mode=False, _real=real):
                calls.append((list(currents), steps, write_mode))
                return _real(node_ids=node_ids, currents=currents, steps=steps,
                             write_mode=write_mode)
            ng.graph.prime_and_propagate = spy
            for preserve in (False, True):
                calls.clear()
                org.cc_pattern_completion_recall(ng, CUE_TEXT, k=2, state={},
                                                 preserve_graph_config=preserve)
                assert len(calls) == 2, calls
                (h_cur, h_steps, h_wm), (s_cur, s_steps, s_wm) = calls
                assert h_wm is False and s_wm is False
                assert s_cur == [0.0], "the sweep injects no stimulus"
                assert s_steps == h_steps, (prop_steps, novelty, preserve, calls)


def test_switch_off_makes_no_extra_sweep(switch):
    switch(False)
    ng = _build_ng()
    calls = []
    real = ng.graph.prime_and_propagate
    ng.graph.prime_and_propagate = lambda **kw: calls.append(kw) or real(**kw)
    org.cc_pattern_completion_recall(ng, CUE_TEXT, k=2)
    assert len(calls) == 1


def _load_base_module():
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", "4e2ed30:cc_ng_organism.py"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("base commit 4e2ed30 not reachable from this checkout")
    spec = importlib.util.spec_from_loader("cc_ng_organism_base_4e2ed30_1131", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(ROOT, "cc_ng_organism.py")  # same dir: same sibling imports
    # dataclasses resolve their module through sys.modules at class creation; register only
    # for the exec, then remove so no other test's module-location guard ever sees it.
    sys.modules[spec.name] = mod
    try:
        exec(compile(src, "cc_ng_organism@4e2ed30", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


_BASE = []


def _base_module():
    if not _BASE:
        _BASE.append(_load_base_module())
    return _BASE[0]


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("preserve", [False, True])
def test_switch_off_is_identical_to_the_base_commit(switch, monkeypatch, k, preserve):
    switch(False)
    base = _base_module()
    monkeypatch.setattr(base, "cc_gsg_rescore", lambda surfaced, *a, **kw: surfaced)
    monkeypatch.setattr(base, "_CC_RECALL_SELECTIVITY", False)
    monkeypatch.setattr(base, "_CC_RECALL_PROP_STEPS", 0)
    now = org.cc_pattern_completion_recall(_build_ng(), CUE_TEXT, k=k,
                                           preserve_graph_config=preserve)
    then = base.cc_pattern_completion_recall(_build_ng(), CUE_TEXT, k=k,
                                             preserve_graph_config=preserve)
    assert now == then
    assert now, "the comparison is not vacuous"


def test_env_defaults_are_off():
    code = ("import cc_ng_organism as o; "
            "print(o._CC_RECALL_EVOKED, o._CC_RECALL_EVOKED_OVERSAMPLE)")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CC_RECALL_EVOKED", "CC_RECALL_EVOKED_OVERSAMPLE")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "False 6", out.stderr[-2000:]
