# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-1) — tests for
#   cc_ng_organism.probation_advances and cc_update_probation's byte-identical refactor
# What: (a) predicate unit tests; (b) test (14): cc_update_probation, with its inline
#   `creation_mode == "ingested"` skip replaced by probation_advances(node), is BYTE-IDENTICAL to
#   the base function (embedded verbatim from NG origin/main b5e47686) over a seeded family of
#   graphs -- ingested / conversational / no-creation_mode / None-metadata / odd-metadata nodes --
#   comparing every node's resulting metadata, excitability, threshold, the returned
#   `graduated` list and (for the raising shapes) the exception class and partial state, over
#   repeated pulses; (c) test (13): the predicate is DEFINED ONCE and CONSULTED, NOT DUPLICATED
#   (a static `ast` test), and neuro_foundation holds no "ingested" literal of its own.
# Why: Josh's ruling (Exec P550 / P552, amended Exec P554 / P556); CC-CALLOSUM-TRUTH §8.13. The
#   orphan sweep (NG-2) reads the same predicate as the thing that advances probation (LAW 4).
# How: no Graph.step(), no NeuroGraphMemory, no checkpoint, no embedder: a real small Graph and the
#   real cc_update_probation only. Deterministic seeds; node order is dict insertion order, so
#   nothing here depends on PYTHONHASHSEED (verified by the 0..31 sweep in the return).
# -------------------
import ast
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import cc_ng_organism as org
from neuro_foundation import Graph

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The base cc_update_probation, VERBATIM from NG origin/main b5e476863cc069a29ec482959b4f9465f2ea4ccf
# (cc_ng_organism.py:2055-2134). It is the reference the refactor must match, so it is a literal copy,
# never a re-derivation.
_BASE_CC_UPDATE_PROBATION_SRC = r'''
def cc_update_probation(graph) -> list:
    """Substrate-level probation graduation -- fades novelty-dampening over
    the probation window and graduates nodes to full excitability. Mirrors
    canonical's _update_probation exactly (neurograph_rpc.py:2145-2169) --
    that function is already parameterized on graph alone, so this is a
    near-verbatim port. Call once per pulse (autosave loop), after any
    conversational deposits for that pulse -- operates on ALL probationary
    nodes, not just ones just deposited.

    Novelty-dampening release is ALWAYS on the timer. Only the "graduated" stamp is
    gated on evidence of firing (#93) -- see the comment at the graduation branch for
    why those two must not be gated together.
    """
    with _cc_mutation_lock(graph):
        graduated = []
        base_threshold = graph.config.get("default_threshold", 1.0)
        for nid, node in list(graph.nodes.items()):
            # #111 -- document nodes belong to the Ingestor's probation sweep
            # (universal_ingestor.py, now scoped to creation_mode == "ingested").
            # Before both sweeps were scoped they walked the same graph, so CC's
            # ingested nodes were decremented twice -- once per prompt via
            # on_message, once per 60s pulse via this function -- burning their
            # window at double rate. One sweeper per probation domain.
            #
            # Deliberately an EXCLUSION, not `== "conversational"`: nodes with no
            # creation_mode (older checkpoints, seeds) must still graduate here
            # rather than be stranded in probation forever.
            #
            # This is where the CC mirror intentionally stops matching canonical
            # neurograph_rpc.py::_update_probation. On Syl the Ingestor sweep is
            # dead code (on_message has no callers there), so _update_probation is
            # the ONLY thing graduating her document nodes and must keep sweeping
            # them. CC-first, back-propagate later: expect these two to differ
            # until canonical is brought over.
            if (node.metadata or {}).get("creation_mode") == "ingested":
                continue
            prob = node.metadata.get("probation_remaining")
            if prob is None:
                continue
            if prob <= 0:
                # Late graduation: a node whose window expired before it ever fired stays
                # eligible. If it fires later it has earned the stamp then -- without this
                # the flag would permanently under-report nodes that entered cognition
                # after their window closed. Already-graduated nodes lack the marker and
                # fall straight through, preserving the original fast path.
                #
                # The gate is INSIDE the marker branch, mirroring the expiry branch below.
                # Gating the branch itself on _CC_CONV_PROBATION_REQUIRE_SPIKE would make
                # the rollback one-way: with the knob off, nodes already stamped
                # probation_expired_unfired would be skipped entirely and stranded at
                # graduated=False forever -- exactly the cohort the knob is flipped to
                # rescue. Rollback must drain the marker, not orphan it.
                if node.metadata.get("probation_expired_unfired"):
                    if not _CC_CONV_PROBATION_REQUIRE_SPIKE or _cc_has_ever_fired(node):
                        node.metadata["graduated"] = True
                        node.metadata.pop("probation_expired_unfired", None)
                        graduated.append(nid)
                continue
            prob -= 1
            node.metadata["probation_remaining"] = prob
            if prob <= 0:
                # Dampening release is unconditional and stays on the timer. Gating it on
                # firing would be a self-reinforcing trap: a never-fired node would keep a
                # permanently boosted threshold, making it even less likely to fire, so it
                # could never earn release.
                node.intrinsic_excitability = 1.0
                node.threshold = base_threshold
                if not _CC_CONV_PROBATION_REQUIRE_SPIKE or _cc_has_ever_fired(node):
                    node.metadata["graduated"] = True
                    graduated.append(nid)
                else:
                    # Aged out without ever firing: dampening lifted, but nothing earned.
                    node.metadata["graduated"] = False
                    node.metadata["probation_expired_unfired"] = True
            else:
                damp = float(node.metadata.get("novelty_dampening", _CC_CONV_NOVELTY_DAMPENING))
                total = float(node.metadata.get("probation_total", _CC_CONV_PROBATION_PERIOD)) or float(_CC_CONV_PROBATION_PERIOD)
                frac = max(0.0, min(1.0, 1.0 - prob / total))
                node.intrinsic_excitability = damp + (1.0 - damp) * frac
        return graduated
'''


def _base_cc_update_probation():
    ns = dict(vars(org))   # the base resolves _cc_mutation_lock / constants / _cc_has_ever_fired here
    exec(compile(_BASE_CC_UPDATE_PROBATION_SRC, "<base cc_update_probation @ b5e47686>", "exec"), ns)
    return ns["cc_update_probation"]


class _N:
    """A bare node-like for the predicate unit tests (the predicate reads only .metadata)."""

    def __init__(self, metadata):
        self.metadata = metadata


# ---------------------------------------------------------------------------
# Predicate unit tests
# ---------------------------------------------------------------------------

def test_predicate_false_for_ingested():
    assert org.probation_advances(_N({"creation_mode": "ingested"})) is False


@pytest.mark.parametrize("meta", [
    {"creation_mode": "conversational"},
    {"creation_mode": "emergent"},
    {"creation_mode": ""},
    {"creation_mode": None},
    {"creation_mode": "Ingested"},      # exact-match semantics, as the base skip had
    {},                                  # no creation_mode: older checkpoints / seeds still decremented
    {"probation_remaining": 5},
    None,                                # `(node.metadata or {})` guard
])
def test_predicate_true_for_everything_else(meta):
    assert org.probation_advances(_N(meta)) is True


def test_predicate_raises_like_the_base_expression_on_non_dict_metadata():
    # The base skip was `(node.metadata or {}).get(...)`: a truthy non-dict raises AttributeError.
    # The refactor must not turn that into a silent True/False.
    with pytest.raises(AttributeError):
        org.probation_advances(_N("not-a-dict"))


def _tiny_graph(**nodes):
    g = Graph()
    for nid, meta in nodes.items():
        g.create_node(node_id=nid, metadata=dict(meta))
    return g


def test_cc_update_probation_consults_the_module_level_predicate(monkeypatch):
    """The decrementer's skip IS the predicate (looked up at call time), not a private copy."""
    g = _tiny_graph(
        conv={"creation_mode": "conversational", "probation_remaining": 5, "probation_total": 5},
        ing={"creation_mode": "ingested", "probation_remaining": 5, "probation_total": 5},
    )
    org.cc_update_probation(g)
    assert g.nodes["conv"].metadata["probation_remaining"] == 4
    assert g.nodes["ing"].metadata["probation_remaining"] == 5     # ingested: not decremented

    monkeypatch.setattr(org, "probation_advances", lambda node: True)
    org.cc_update_probation(g)
    assert g.nodes["ing"].metadata["probation_remaining"] == 4     # predicate True => decremented
    assert g.nodes["conv"].metadata["probation_remaining"] == 3

    monkeypatch.setattr(org, "probation_advances", lambda node: False)
    org.cc_update_probation(g)
    assert g.nodes["conv"].metadata["probation_remaining"] == 3    # predicate False => skipped
    assert g.nodes["ing"].metadata["probation_remaining"] == 4


# ---------------------------------------------------------------------------
# (14) cc_update_probation byte-identical to the base over a seeded family
# ---------------------------------------------------------------------------

_ABSENT = object()
_MODES = ["ingested", "conversational", None, "emergent", "weird"]            # None => key absent
_PROBS_CLEAN = [_ABSENT, None, 0, 1, 2, 3, 5, 10, -1, 2.5]
_PROBS_ODD = _PROBS_CLEAN + ["3", "x", float("nan"), True, float("inf"), 0.0]
_N_NODES = 24
_PULSES = 14


def _build(seed, odd):
    rng = random.Random(seed)
    g = Graph()
    probs = _PROBS_ODD if odd else _PROBS_CLEAN
    for i in range(_N_NODES):
        meta = {}
        mode = rng.choice(_MODES)
        if mode is not None:
            meta["creation_mode"] = mode
        prob = rng.choice(probs)
        if prob is not _ABSENT:
            meta["probation_remaining"] = prob
        if rng.random() < 0.6:
            meta["probation_total"] = rng.choice([3, 5, 10])
        if rng.random() < 0.4:
            meta["novelty_dampening"] = rng.choice([0.1, 0.3, 0.5])
        if rng.random() < 0.15:
            meta["probation_expired_unfired"] = True
        if rng.random() < 0.1:
            meta["graduated"] = rng.choice([True, False])
        node = g.create_node(node_id="n%02d" % i, metadata=meta)
        if rng.random() < 0.5:
            node.spike_history.append(float(rng.randint(1, 50)))     # has genuinely fired
        node.intrinsic_excitability = rng.choice([0.3, 0.5, 1.0])
        node.threshold = rng.choice([0.85, 1.05, 1.2])
    if odd:
        for nid in rng.sample(sorted(g.nodes), 2):
            if rng.random() < 0.5:
                g.nodes[nid].metadata = None                         # the None-metadata shape
    return g


def _snapshot(g):
    out = []
    for nid, node in g.nodes.items():
        md = node.metadata
        mdr = "None" if md is None else repr(sorted(md.items(), key=lambda kv: kv[0]))
        out.append((nid, mdr, repr(node.intrinsic_excitability), repr(node.threshold)))
    return out


def _run(fn, g):
    try:
        return ("ok", fn(g))
    except Exception as exc:   # noqa: BLE001 -- the exception CLASS is part of "identical"
        return ("raised", type(exc).__name__)


def _compare_family(seeds, odd):
    base = _base_cc_update_probation()
    raised = graduated = 0
    for seed in seeds:
        gb, gn = _build(seed, odd), _build(seed, odd)
        assert _snapshot(gb) == _snapshot(gn), "builder is not deterministic (seed %d)" % seed
        for pulse in range(_PULSES):
            rb, rn = _run(base, gb), _run(org.cc_update_probation, gn)
            assert rb == rn, "seed %d pulse %d: result %r != %r" % (seed, pulse, rb, rn)
            assert _snapshot(gb) == _snapshot(gn), "seed %d pulse %d: state diverged" % (seed, pulse)
            if rb[0] == "raised":
                raised += 1
                break      # both aborted mid-loop identically; partial state already compared
            graduated += len(rb[1])
    return raised, graduated


def test_cc_update_probation_byte_identical_clean_family():
    raised, graduated = _compare_family(range(150), odd=False)
    assert raised == 0
    assert graduated > 100          # the family really exercises graduation (not vacuous)


def test_cc_update_probation_byte_identical_odd_family():
    raised, graduated = _compare_family(range(150), odd=True)
    assert raised > 20              # the raising shapes (str probation / None metadata) really occur
    # Most odd graphs abort on their first raising node (identically in both), so graduation is
    # rarer here than in the clean family. Measured 24 over 150 seeds; the floor only guards
    # against the family silently degenerating to "everything raises before anything graduates".
    assert graduated > 10


# ---------------------------------------------------------------------------
# (13) the predicate is DEFINED ONCE and CONSULTED, NOT DUPLICATED (static)
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", "tests", "data", "Defunct-Historical", "__pycache__", "docs"}


def _repo_py_files():
    for root, dirs, files in os.walk(_REPO):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _parse(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def test_probation_advances_defined_exactly_once_in_cc_ng_organism():
    defs = []
    for path in _repo_py_files():
        for n in ast.walk(_parse(path)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "probation_advances":
                defs.append((os.path.relpath(path, _REPO), n.lineno))
    assert len(defs) == 1, defs
    assert defs[0][0] == "cc_ng_organism.py"
    top = [n for n in _parse(os.path.join(_REPO, "cc_ng_organism.py")).body
           if isinstance(n, ast.FunctionDef) and n.name == "probation_advances"]
    assert len(top) == 1            # module-level, and not rebound by a second module-level def


def test_cc_update_probation_references_the_predicate_and_holds_no_literal():
    tree = _parse(os.path.join(_REPO, "cc_ng_organism.py"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cc_update_probation")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name) and c.func.id == "probation_advances"]
    assert len(calls) == 1
    literals = [c for c in ast.walk(fn) if isinstance(c, ast.Constant) and c.value == "ingested"]
    assert literals == []


def test_the_ingested_literal_lives_only_in_the_predicate_within_cc_ng_organism():
    tree = _parse(os.path.join(_REPO, "cc_ng_organism.py"))
    holders = []
    for top in tree.body:
        for c in ast.walk(top):
            if isinstance(c, ast.Constant) and c.value == "ingested":
                holders.append(getattr(top, "name", "<module-level>"))
    assert holders == ["probation_advances"], holders


def test_neuro_foundation_holds_no_ingested_literal_and_imports_no_cc_module():
    tree = _parse(os.path.join(_REPO, "neuro_foundation.py"))
    for c in ast.walk(tree):
        if isinstance(c, ast.Constant):
            assert c.value != "ingested"
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert not any(a.name.startswith("cc_") for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert not (n.module or "").startswith("cc_")
