# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-3 / round 2) — Josh's ruling (Exec P550 / P552; Exec P561; Exec P562)
# What: RE-KEYED ONLY, no assertion changed or weakened. The round-1 predicate `cc_ng_organism.probation_advances` no longer exists (renamed
#   with no alias: its population test is now `probation_population`, its window logic is `fair_chance_window_open`). The three places that
#   registered the REAL round-1 predicate now register `org.probation_population`, which is byte-for-byte the same test, so every round-1
#   expectation holds unchanged against the round-1 sweep body that is still committed at NG-3. The sweep body and this file are re-keyed
#   again, with a change of meaning, in NG-4 (the protected body becomes host-agnostic).
# Why: each commit must be GREEN on its own and the rename rule is one name everywhere.
# [2026-10-02] (round 1, NG-2) Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-2) — tests (1)-(12) for the
#   orphan sweep's probation exemption (neuro_foundation.Graph._collect_orphan_nodes)
# What: a REAL small Graph, the real step() / real _collect_orphan_nodes / real cc_update_probation / the REAL
#   cc_ng_organism.probation_advances registered as graph._probation_advances. (1) an unbound node past grace with
#   an open window survives several real steps; (2) once the window reaches 0 UNWIRED (driven by the real
#   cc_update_probation, never by editing metadata) the next sweep takes it; (3) a node that wires during its window
#   (a real hyperedge; and the engine's own co-firing sprout) survives after the window ends; (4) identity-protected
#   nodes are unaffected either way; (5) no probation key => today's sweep, identical to the base function (embedded
#   VERBATIM from NG origin/main b5e47686) over a seeded family; (6) odd shapes (None/str/negative/zero/NaN/bool/inf/
#   falsy-non-dict metadata) are NOT spared and never reach the predicate; (7) grace unchanged; (9) an `ingested` node
#   with an open window is STILL swept; (10) a conversational node in the same state is spared; (11) NO registration
#   => exactly today's sweep (Syl's case), over a seeded family that includes open windows; (12) a raising predicate =>
#   that node is NOT spared and ONE WARNING per sweep with a count, class names only. Plus: the predicate is consulted
#   LAST (never for bound / young / protected nodes), the exemption only ever SPARES (never sweeps more than today),
#   and the function carries no cc_-named reference. (8) the pre-existing orphan/grace tests are run in the return.
# Why: Josh's ruling (Exec P550 / P552, amended P554 / P556); CC-CALLOSUM-TRUTH §8.13.
# How: scratch Graphs only; no checkpoint, no NeuroGraphMemory, no embedder. Deterministic seeds. The sweep builds its
#   candidate list by iterating self.nodes (a dict: insertion order), so nothing here depends on PYTHONHASHSEED (the
#   0..31 sweep is in the return).
# -------------------
import ast
import logging
import math
import os
import random
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import cc_ng_organism as org
from neuro_foundation import Graph

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRACE = 25   # DEFAULT_CONFIG["orphan_node_grace_period"]

# The base _collect_orphan_nodes, VERBATIM from NG origin/main b5e476863cc069a29ec482959b4f9465f2ea4ccf
# (neuro_foundation.py:3574-3611). The reference "today's sweep" must match: a literal copy, never a re-derivation.
_BASE_COLLECT_SRC = r'''
    def _collect_orphan_nodes(self) -> int:
        """Remove nodes with no synapses and no hyperedge membership.

        Called after _prune_synapses() so freshly-disconnected nodes are
        collected in the same structural-plasticity step. The full SNN has
        no max_nodes cap — without this, orphans accumulate without bound
        as synapses are pruned over the graph's lifetime.

        Honors orphan_node_grace_period (#258): newly-created nodes get a
        window for canonical mechanisms (STDP via spreading activation
        through existing synapses, sprouting via co-firing detection) to
        wire them before sweep. Without grace, empty-substrate bootstrap
        fails — the very first deposit gets swept on the next step()
        because no co-firing partners exist yet to anchor synapses. Mature
        substrates (Syl) are unaffected: new nodes already get wired via
        spreading activation within the same step() before the orphan
        check runs at step 8, so grace passes but isn't load-bearing.
        Restored nodes from older msgpacks default to creation_time=0
        and thus age = full current timestep, well past grace — same
        sweep behavior as before this patch.
        """
        grace = self.config.get("orphan_node_grace_period", 0)
        orphans = [
            nid for nid in self.nodes
            if not self._outgoing.get(nid)
            and not self._incoming.get(nid)
            and not self._node_hyperedges.get(nid)
            and (self.timestep - self.nodes[nid].creation_time) > grace
            and not self._is_identity_protected(nid)  # #spine: never sweep her authored self
        ]
        removed = 0
        for nid in orphans:
            if nid in self.nodes:
                self.remove_node(nid)
                removed += 1
        if removed:
            self._emit("nodes_collected", count=removed, timestep=self.timestep)
        return removed
'''


def _base_collect():
    ns = {"logger": logging.getLogger("neuro_foundation.p552_base")}
    exec(compile(textwrap.dedent(_BASE_COLLECT_SRC), "<base _collect_orphan_nodes @ b5e47686>", "exec"), ns)
    return ns["_collect_orphan_nodes"]


def _conv(prob=5, **extra):
    meta = {"creation_mode": "conversational", "probation_remaining": prob, "probation_total": 10}
    meta.update(extra)
    return meta


def _graph(timestep=10_000, register=True):
    g = Graph()
    g.timestep = timestep
    if register:
        g._probation_advances = org.probation_population
    return g


def _node(g, nid, meta=None, age=GRACE + 100):
    n = g.create_node(node_id=nid, metadata=dict(meta) if meta is not None else {})
    n.creation_time = g.timestep - age
    return n


def _snap(g):
    return (sorted(g.nodes), len(g.synapses), len(g.hyperedges))


# ---------------------------------------------------------------------------
# attribute plumbing: the host can register on a plain Graph instance
# ---------------------------------------------------------------------------

def test_graph_accepts_a_plain_attribute_and_has_none_by_default():
    assert not hasattr(Graph(), "_probation_advances")                  # nothing registered by default
    assert "__slots__" not in Graph.__dict__
    assert not isinstance(getattr(Graph, "_probation_advances", None), property)
    g = Graph()
    g._probation_advances = org.probation_population
    assert g._probation_advances is org.probation_population            # the function object ITSELF


# ---------------------------------------------------------------------------
# (1) an unbound node past grace with an open window SURVIVES several real steps
# ---------------------------------------------------------------------------

def test_1_open_window_survives_real_steps_past_grace():
    g = _graph(timestep=0)
    _node(g, "conv", _conv(prob=5), age=0)
    _node(g, "ctrl", {"creation_mode": "conversational"}, age=0)        # no key: today's sweep
    for _ in range(GRACE + 40):
        g.step()
    assert g.timestep > GRACE + 30
    assert "ctrl" not in g.nodes          # control: the real step() DID reach the sweep and took it
    assert "conv" in g.nodes              # the open window protected it, across many sweeps past grace


# ---------------------------------------------------------------------------
# (2) the window closes on the REAL cc_update_probation; the next sweep then takes it
# ---------------------------------------------------------------------------

def test_2_window_closes_via_real_cc_update_probation_then_swept():
    g = _graph()
    _node(g, "conv", _conv(prob=3))
    for expected_remaining in (2, 1, 0):
        g.step()
        assert "conv" in g.nodes                                           # window open at this sweep
        org.cc_update_probation(g)                                         # the autonomic pulse
        assert g.nodes["conv"].metadata["probation_remaining"] == expected_remaining
    g.step()                                                               # window closed, still unwired
    assert "conv" not in g.nodes


# ---------------------------------------------------------------------------
# (3) a node that WIRES during its window survives after the window ends
# ---------------------------------------------------------------------------

def test_3_wired_by_a_real_hyperedge_during_the_window_survives_after_it():
    g = _graph()
    _node(g, "X", _conv(prob=3))
    _node(g, "Y", _conv(prob=3))
    _node(g, "W", _conv(prob=3))                                           # never wired
    g.step()
    g.create_hyperedge({"X", "Y"})                                         # a real membership, mid-window
    for _ in range(5):
        org.cc_update_probation(g)
        g.step()
    assert g.nodes["X"].metadata["probation_remaining"] == 0
    assert "X" in g.nodes and "Y" in g.nodes                              # bound: survives after the window
    assert "W" not in g.nodes                                              # unwired twin: swept at expiry


def test_3b_wired_by_the_engines_own_cofiring_sprout_survives_after_the_window():
    g = _graph()
    _node(g, "A", _conv(prob=4))
    _node(g, "B", _conv(prob=4))
    _node(g, "W", _conv(prob=4))                                           # never fires, never wired
    # Real dynamics: A then B fire on consecutive steps; STDP/sprouting wires them (no injected synapse).
    g.stimulate("A", 5.0)
    g.step()
    g.stimulate("B", 5.0)
    g.step()
    assert g._find_synapse("A", "B") is not None or g._find_synapse("B", "A") is not None, \
        "precondition: the engine's own co-firing sprouted a synapse"
    for _ in range(6):
        org.cc_update_probation(g)
        g.step()
    assert g.nodes["A"].metadata["probation_remaining"] == 0
    assert "A" in g.nodes and "B" in g.nodes
    assert "W" not in g.nodes


# ---------------------------------------------------------------------------
# (4) identity-protected nodes are unaffected either way
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("register", [True, False])
@pytest.mark.parametrize("prob", [5, 0, -1, None, "x"])
def test_4_identity_protected_unaffected(register, prob):
    g = _graph(register=register)
    calls = []
    if register:
        g._probation_advances = lambda n: calls.append(n) or True
    meta_c = {"constitutional": True}
    meta_w = {"provenance": "cc_authored"}
    if prob is not None:
        meta_c["probation_remaining"] = prob
        meta_w["probation_remaining"] = prob
    _node(g, "const", meta_c)
    _node(g, "want", meta_w)
    _node(g, "plain", {})
    g._collect_orphan_nodes()
    assert "const" in g.nodes and "want" in g.nodes                       # protected: never swept
    assert "plain" not in g.nodes
    assert calls == []                                                     # protection is decided before the term


# ---------------------------------------------------------------------------
# (5) no probation key => today's sweep, identical to the base function (even with the predicate registered)
# (11) no registration => today's sweep, identical to the base function (Syl's case), open windows included
# ---------------------------------------------------------------------------

_PROBS_ALL = [0, 1, 3, 10, -1, 2.5, None, "4", float("nan"), True, False, float("inf"), -float("inf"), 0.0]


def _family(seed, probs, register):
    rng = random.Random(seed)
    g = _graph(timestep=500, register=register)
    ids = []
    for i in range(30):
        nid = "n%02d" % i
        ids.append(nid)
        meta = {}
        if rng.random() < 0.3:
            meta["creation_mode"] = rng.choice(["ingested", "conversational", "emergent"])
        if probs is not None and rng.random() < 0.7:
            meta["probation_remaining"] = rng.choice(probs)
        r = rng.random()
        if r < 0.08:
            meta["constitutional"] = True
        elif r < 0.16:
            meta["provenance"] = rng.choice(["syl_authored", "cc_authored", "cc_emergent"])
        _node(g, nid, meta, age=rng.choice([0, 5, GRACE, GRACE + 1, 100, 400]))
    seen = set()
    for _ in range(rng.randint(3, 10)):
        a, b = rng.sample(ids, 2)
        if (a, b) not in seen:
            seen.add((a, b))
            g.create_synapse(a, b, weight=0.2)
    for _ in range(rng.randint(0, 3)):
        g.create_hyperedge(set(rng.sample(ids, 3)))
    return g


def _same_as_base(seed, probs, register):
    base = _base_collect()
    gb, gn = _family(seed, probs, register=False), _family(seed, probs, register=register)
    assert _snap(gb) == _snap(gn)
    rb, rn = base(gb), gn._collect_orphan_nodes()
    assert rb == rn, "seed %d: removed %d != %d" % (seed, rb, rn)
    assert _snap(gb) == _snap(gn), "seed %d: graph state diverged" % seed
    return rb


def test_5_no_probation_key_is_todays_sweep_even_when_the_predicate_is_registered():
    removed = sum(_same_as_base(s, None, register=True) for s in range(120))
    assert removed > 100                                                   # the family really sweeps (not vacuous)


def test_11_no_registration_is_exactly_todays_sweep_open_windows_included():
    removed = sum(_same_as_base(s, _PROBS_ALL, register=False) for s in range(120))
    assert removed > 100


def test_the_exemption_only_ever_spares_and_does_something_when_registered():
    """With the predicate registered a node today's sweep removes may be KEPT, but no node today's sweep keeps is ever
    removed (monotone), and over the family it does spare something (non-vacuity of the registered path)."""
    base = _base_collect()
    spared_total = 0
    for seed in range(120):
        gb, gn = _family(seed, _PROBS_ALL, register=False), _family(seed, _PROBS_ALL, register=True)
        base(gb)
        gn._collect_orphan_nodes()
        assert set(gb.nodes) <= set(gn.nodes), "seed %d: the exemption removed a node today's sweep keeps" % seed
        spared_total += len(set(gn.nodes) - set(gb.nodes))
    assert spared_total > 20


# ---------------------------------------------------------------------------
# (6) odd shapes are NOT spared and never reach the predicate; (6b) numeric > 0 IS spared
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prob", [None, "5", "", "abc", [], [3], {}, -1, -0.5, 0, 0.0, False, True,
                                  float("nan"), float("inf"), -float("inf")])
def test_6_odd_probation_values_are_not_spared_and_never_reach_the_predicate(prob):
    g = _graph()
    calls = []
    g._probation_advances = lambda n: calls.append(n) or True
    _node(g, "odd", {"probation_remaining": prob})
    g._collect_orphan_nodes()
    assert "odd" not in g.nodes
    assert calls == []


@pytest.mark.parametrize("prob", [1, 3, 0.5, 2.5, 10 ** 12, 2 ** 70])
def test_6b_numeric_greater_than_zero_is_spared_when_the_predicate_says_so(prob):
    g = _graph()
    g._probation_advances = lambda n: True
    _node(g, "ok", {"probation_remaining": prob})
    g._collect_orphan_nodes()
    assert "ok" in g.nodes


@pytest.mark.parametrize("meta", [None, [], "", 0])
def test_6c_falsy_non_dict_metadata_is_swept(meta):
    g = _graph()
    calls = []
    g._probation_advances = lambda n: calls.append(n) or True
    n = _node(g, "odd", {})
    n.metadata = meta
    g._collect_orphan_nodes()
    assert "odd" not in g.nodes
    assert calls == []


def test_6d_truthy_non_dict_metadata_fails_exactly_as_the_base_does_before_the_term():
    """A truthy non-dict metadata already raises inside _is_identity_protected (a pre-existing shape the change does
    not touch). The new function must raise the SAME way, not swallow it and not turn it into a spare."""
    base = _base_collect()
    gb, gn = _graph(register=False), _graph()
    for g in (gb, gn):
        n = _node(g, "bad", {})
        n.metadata = "not-a-dict"
    with pytest.raises(AttributeError):
        base(gb)
    with pytest.raises(AttributeError):
        gn._collect_orphan_nodes()


# ---------------------------------------------------------------------------
# (7) grace is unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age,swept", [(0, False), (GRACE - 1, False), (GRACE, False), (GRACE + 1, True), (400, True)])
def test_7_grace_unchanged_for_a_node_with_no_probation_key(age, swept):
    g = _graph()
    _node(g, "n", {}, age=age)
    g._collect_orphan_nodes()
    assert ("n" not in g.nodes) is swept


@pytest.mark.parametrize("age", [0, GRACE, GRACE + 1, 400])
def test_7b_an_open_window_spares_at_every_age_and_grace_still_spares_a_young_node(age):
    g = _graph()
    calls = []
    g._probation_advances = lambda n: calls.append(n.metadata.get("probation_remaining")) or True
    _node(g, "n", _conv(prob=3), age=age)
    g._collect_orphan_nodes()
    assert "n" in g.nodes
    # young (age <= grace): spared by GRACE alone, the predicate is never consulted; old: spared by the window
    assert calls == ([] if age <= GRACE else [3])


# ---------------------------------------------------------------------------
# (9) ingested + open window is STILL swept (the leak case); (10) conversational same state is spared
# ---------------------------------------------------------------------------

def test_9_ingested_node_with_an_open_window_is_still_swept_with_the_real_predicate():
    g = _graph()
    _node(g, "ing", {"creation_mode": "ingested", "probation_remaining": 10, "probation_total": 10})
    g._collect_orphan_nodes()
    assert "ing" not in g.nodes


def test_10_conversational_node_in_the_same_state_is_spared():
    g = _graph()
    _node(g, "conv", {"creation_mode": "conversational", "probation_remaining": 10, "probation_total": 10})
    _node(g, "bare", {"probation_remaining": 10, "probation_total": 10})   # no creation_mode: still decremented => spared
    g._collect_orphan_nodes()
    assert "conv" in g.nodes and "bare" in g.nodes
    assert g._probation_advances is org.probation_population               # the REAL (population) predicate, not a stub


# ---------------------------------------------------------------------------
# (12) a raising predicate => that node is NOT spared, ONE WARNING per sweep with a count
# ---------------------------------------------------------------------------

_SECRET = "secret-detail-must-never-be-logged-xyz"


def _warnings(caplog):
    return [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING]


def test_12_raising_predicate_node_not_spared_one_warning_with_count_and_class_only(caplog):
    g = _graph()

    def boom(node):
        raise RuntimeError(_SECRET)

    g._probation_advances = boom
    for i in range(3):
        _node(g, "c%d" % i, _conv(prob=5))
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        removed = g._collect_orphan_nodes()
    assert removed == 3 and not g.nodes                                    # fail toward today: all swept
    ws = _warnings(caplog)
    assert len(ws) == 1                                                    # ONE per sweep, not one per node
    msg = ws[0].getMessage()
    assert "3" in msg and "RuntimeError" in msg                            # a count and the class name
    assert _SECRET not in msg and _SECRET not in str(ws[0].args)           # never str(exc)


def test_12b_one_warning_per_sweep_not_per_process(caplog):
    g = _graph()
    g._probation_advances = lambda n: (_ for _ in ()).throw(ValueError(_SECRET))
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        _node(g, "a", _conv(prob=5))
        g._collect_orphan_nodes()
        _node(g, "b", _conv(prob=5))
        g._collect_orphan_nodes()
    assert len(_warnings(caplog)) == 2


def test_12c_mixed_raise_and_true_spares_the_true_one_and_counts_only_the_raise(caplog):
    g = _graph()

    def sometimes(node):
        if node.metadata.get("who") == "bad":
            raise KeyError(_SECRET)
        return True

    g._probation_advances = sometimes
    _node(g, "good", _conv(prob=5, who="good"))
    _node(g, "bad", _conv(prob=5, who="bad"))
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
    assert "good" in g.nodes and "bad" not in g.nodes
    ws = _warnings(caplog)
    assert len(ws) == 1 and "1 node" in ws[0].getMessage() and "KeyError" in ws[0].getMessage()
    assert _SECRET not in ws[0].getMessage()


def test_12d_no_warning_when_the_predicate_does_not_raise(caplog):
    g = _graph()
    _node(g, "conv", _conv(prob=5))
    _node(g, "ing", {"creation_mode": "ingested", "probation_remaining": 5})
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
    assert _warnings(caplog) == []


def test_12e_a_non_callable_registration_fails_toward_todays_sweep_with_a_warning(caplog):
    g = _graph()
    g._probation_advances = 7                                              # not callable: a host bug
    _node(g, "conv", _conv(prob=5))
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
    assert "conv" not in g.nodes
    assert len(_warnings(caplog)) == 1 and "TypeError" in _warnings(caplog)[0].getMessage()


# ---------------------------------------------------------------------------
# the predicate is consulted LAST: never for bound / young / identity-protected nodes
# ---------------------------------------------------------------------------

def test_the_predicate_is_consulted_only_for_unbound_old_unprotected_numeric_open_windows():
    g = _graph()
    seen = []
    g._probation_advances = lambda n: seen.append(n) or True
    _node(g, "bound_a", _conv(prob=5))
    _node(g, "bound_b", _conv(prob=5))
    g.create_synapse("bound_a", "bound_b", weight=0.2)                     # structurally bound
    _node(g, "hyper_a", _conv(prob=5))
    _node(g, "hyper_b", _conv(prob=5))
    g.create_hyperedge({"hyper_a", "hyper_b"})                             # structurally bound
    _node(g, "young", _conv(prob=5), age=3)                                # inside grace
    _node(g, "const", _conv(prob=5, constitutional=True))                  # identity-protected
    _node(g, "target", _conv(prob=5))                                      # the only candidate
    _node(g, "closed", _conv(prob=0))                                      # window closed: never reaches pred
    g._collect_orphan_nodes()
    assert len(seen) == 1
    assert seen[0] is g.nodes["target"]
    assert "closed" not in g.nodes
    for keep in ("bound_a", "bound_b", "hyper_a", "hyper_b", "young", "const", "target"):
        assert keep in g.nodes


def test_a_bound_node_with_an_expired_window_is_never_affected():
    g = _graph()
    _node(g, "a", _conv(prob=0))
    _node(g, "b", _conv(prob=-3))
    g.create_synapse("a", "b", weight=0.2)
    g._collect_orphan_nodes()
    assert "a" in g.nodes and "b" in g.nodes


def test_the_sweep_is_not_disabled_by_an_open_window_on_some_other_node():
    g = _graph()
    _node(g, "open", _conv(prob=5))
    _node(g, "plain", {})
    _node(g, "closed", _conv(prob=0))
    removed = g._collect_orphan_nodes()
    assert removed == 2 and set(g.nodes) == {"open"}


# ---------------------------------------------------------------------------
# static: ONE function, the getattr read, no cc_-named reference, no 'ingested' literal
# ---------------------------------------------------------------------------

def _sweep_fn():
    with open(os.path.join(_REPO, "neuro_foundation.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Graph":
            for n in cls.body:
                if isinstance(n, ast.FunctionDef) and n.name == "_collect_orphan_nodes":
                    return n
    raise AssertionError("Graph._collect_orphan_nodes not found")


def test_static_the_sweep_reads_the_registered_predicate_via_getattr_and_names_no_cc_module():
    fn = _sweep_fn()
    reads = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == "getattr" and len(c.args) == 3
             and isinstance(c.args[1], ast.Constant) and c.args[1].value == "_probation_advances"
             and isinstance(c.args[2], ast.Constant) and c.args[2].value is None]
    assert len(reads) == 1
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            assert not n.id.startswith("cc_")
        if isinstance(n, ast.Attribute):
            assert not n.attr.startswith("cc_")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            raise AssertionError("no import may live in the sweep body")
    body = fn.body[1:]                                                      # [0] is the docstring
    for stmt in body:
        for c in ast.walk(stmt):
            if isinstance(c, ast.Constant) and isinstance(c.value, str):
                assert c.value != "ingested"                               # the sweep holds no literal of its own
