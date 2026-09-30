"""#808: the unbounded `surface_wants_for_graph` twin is retired into `surface_wants`.

# ---- Changelog ----
# [2026-09-30] Claude Code (Sonnet 5.5, Z12 worker), lane unbounded-want-twin-808
# What: tests for the delegate. Oversized / backtick-preceded / nested spans are
#   not materialized through the per-deposit twin; a well-formed want keeps its
#   exact `want::` id, node metadata, synapse and return dicts (golden comparison
#   against the BASE twin loaded from `git show e4ebf982:cc_ng_organism.py`);
#   idempotence; return shape for existing callers; Choice Clause floor; skip
#   counting and log levels; the documented create_node behaviour difference.
# Why: handoffs/z12-want-twin-808/plan-001.md section 7. Exec P379: prove the code
#   under test is the worktree copy, not ~/NeuroGraph on PYTHONPATH.
# How: fake in-memory graph/vdb only. No checkpoint, no ~/.claude/plugins/neurograph,
#   no Syl path. The base module is exec'd from git into a throwaway module name.
# -------------------
"""
import logging
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

WORKTREE = Path(__file__).resolve().parents[1]
BASE_REV = "e4ebf982b1989fd9066d610b94853bc68bf70d37"

# Snapshot BEFORE cc_ng_organism is imported here: which NG modules were already
# loaded at session start, and from where.
_NG_STEMS = sorted(p.stem for p in WORKTREE.glob("*.py"))
_PRE_IMPORT_NG = {n: str(getattr(m, "__file__", None))
                  for n, m in sys.modules.items() if n in _NG_STEMS}

import cc_ng_organism as org  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────

class _FakeNode:
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})
        self.creation_time = 0.0


class _FakeGraph:
    def __init__(self, create_node_raises=None):
        self.nodes = {}
        self.synapses = []
        self._step_lock = threading.RLock()   # the canonical lock the want functions take
        self._raises = create_node_raises

    def create_node(self, node_id, metadata=None):
        if self._raises is not None:
            raise self._raises
        n = _FakeNode(metadata)
        self.nodes[node_id] = n
        return n

    def create_synapse(self, a, b, weight=0.0):
        self.synapses.append((a, b, weight))


class _FakeVDB:
    def __init__(self, content):
        self.content = content


SRC = "cc:conv::src1"


def _world(content_by_node, **graph_kw):
    """content_by_node: {node_id: text}. Every node is conversational."""
    g = _FakeGraph(**graph_kw)
    for nid in content_by_node:
        g.nodes[nid] = _FakeNode({"creation_mode": "conversational"})
    return g, _FakeVDB(dict(content_by_node))


def _want_nodes(g):
    return {nid: dict(n.metadata) for nid, n in g.nodes.items()
            if n.metadata.get("kind") == "want"}


@pytest.fixture(scope="module")
def base():
    """The BASE (e4ebf982) cc_ng_organism, exec'd from git into a throwaway module."""
    src = subprocess.run(
        ["git", "-C", str(WORKTREE), "show", f"{BASE_REV}:cc_ng_organism.py"],
        capture_output=True, text=True, check=True).stdout
    name = "cc_ng_organism_base_808"
    mod = types.ModuleType(name)
    mod.__file__ = f"<git:{BASE_REV[:8]}:cc_ng_organism.py>"
    sys.modules[name] = mod          # @dataclass looks the defining module up here
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    yield mod
    sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _reset_skip_state(monkeypatch):
    monkeypatch.setattr(org, "_want_skip_last", 0)


# ── Exec P379: the code under test is the worktree copy ──────────────────

def test_module_under_test_is_the_worktree_copy(capsys):
    resolved = Path(org.__file__).resolve()
    post = {n: str(getattr(m, "__file__", None))
            for n, m in sys.modules.items() if n in _NG_STEMS}
    with capsys.disabled():
        print("\n[808 P379] worktree              :", WORKTREE)
        print("[808 P379] cc_ng_organism.__file__:", resolved)
        print("[808 P379] NG modules in sys.modules BEFORE import:", _PRE_IMPORT_NG or "none")
        print("[808 P379] neurograph_rpc in sys.modules now     :", "neurograph_rpc" in sys.modules)
        print("[808 P379] NG modules in sys.modules now         :", post)
    assert resolved == WORKTREE / "cc_ng_organism.py", (
        "cc_ng_organism resolved OUTSIDE the worktree (PYTHONPATH shadowing): %s" % resolved)
    for n, f in {**_PRE_IMPORT_NG, **post}.items():
        assert f != "None" and Path(f).resolve().is_relative_to(WORKTREE), (
            "NG module %s loaded from outside the worktree: %s" % (n, f))
    assert "neurograph_rpc" not in sys.modules or (
        Path(sys.modules["neurograph_rpc"].__file__).resolve() == WORKTREE / "neurograph_rpc.py")


# ── the defect, and its fix ──────────────────────────────────────────────

OVERSIZED = "[WANT]" + ("x" * (org.WANT_MAX_CHARS + 50)) + "[/WANT]"
DOC_MENTION = ("The reaction loop dispatches on markers:\n- `[WANT]` -> write to the wants "
               "register\n" + "filler discussion of the architecture. " * 40 +
               "\nand the closing `[/WANT]` ends the span.\n")


def test_base_twin_really_materializes_the_oversized_span(base):
    """Proves the harness compares against the real defect, not a strawman."""
    g, vdb = _world({SRC: OVERSIZED})
    base.surface_wants_for_graph(g, vdb)
    lens = [len(m["want_text"]) for m in _want_nodes(g).values()]
    assert lens and max(lens) > org.WANT_MAX_CHARS


def test_twin_does_not_materialize_oversized_span():
    g, vdb = _world({SRC: OVERSIZED})
    assert org.surface_wants_for_graph(g, vdb) == []
    assert _want_nodes(g) == {}


def test_twin_does_not_materialize_backtick_documentation_marker():
    g, vdb = _world({SRC: "see `[WANT]write it down[/WANT]` for the syntax"})
    assert org.surface_wants_for_graph(g, vdb) == []
    assert _want_nodes(g) == {}


def test_twin_does_not_materialize_prose_swallowed_span():
    g, vdb = _world({SRC: DOC_MENTION})
    assert org.surface_wants_for_graph(g, vdb) == []
    assert _want_nodes(g) == {}


def test_twin_does_not_materialize_nested_marker_span():
    g, vdb = _world({SRC: "[WANT] outer [WANT] inner [/WANT]"})
    assert org.surface_wants_for_graph(g, vdb) == []
    assert _want_nodes(g) == {}


def test_twin_skips_are_the_same_skips_as_the_guarded_function():
    contents = {"cc:conv::a": OVERSIZED, "cc:conv::b": "`[WANT]x[/WANT]`",
                "cc:conv::c": "[WANT] a [WANT] b [/WANT]", "cc:conv::d": DOC_MENTION}
    g1, v1 = _world(contents)
    g2, v2 = _world(contents)
    assert org.surface_wants_for_graph(g1, v1) == org.surface_wants(g2, v2) == []
    assert _want_nodes(g1) == _want_nodes(g2) == {}


# ── golden: a well-formed want is unchanged ──────────────────────────────

WELL_FORMED = {
    "cc:conv::a": "thinking out loud [WANT] follow up on the numpy/scipy conflict later [/WANT] anyway",
    "cc:conv::b": "[WANT]first thing[/WANT] noise [WANT]\nsecond\nthing with éè unicode\n[/WANT]",
    "cc:conv::c": "no markers at all",
}


def test_well_formed_want_golden_against_base_twin(base):
    """Same input through BASE code and BRANCH code: identical ids, metadata,
    synapses and return dicts (order included)."""
    gb, vb = _world(WELL_FORMED)
    gn, vn = _world(WELL_FORMED)
    ret_base = base.surface_wants_for_graph(gb, vb)
    ret_new = org.surface_wants_for_graph(gn, vn)
    assert ret_new == ret_base
    assert len(ret_new) == 3
    assert all(w["id"].startswith("want::") and len(w["id"]) == len("want::") + 16 for w in ret_new)
    assert _want_nodes(gn) == _want_nodes(gb)
    assert gn.synapses == gb.synapses
    assert sorted(gn.nodes) == sorted(gb.nodes)
    for nid in (w["id"] for w in ret_new):
        assert gn.nodes[nid].metadata["provenance"] == "cc_authored"
        assert gn.nodes[nid].metadata["want_state"] == "open"
        assert gn.nodes[nid].metadata["creation_mode"] == "conversational"


def test_golden_survives_a_second_call_and_prior_open_wants(base):
    """Existing want nodes come back in the return exactly as the base twin returned them."""
    gb, vb = _world(WELL_FORMED)
    gn, vn = _world(WELL_FORMED)
    base.surface_wants_for_graph(gb, vb)
    org.surface_wants_for_graph(gn, vn)
    assert org.surface_wants_for_graph(gn, vn) == base.surface_wants_for_graph(gb, vb)


def test_idempotent_second_call_creates_nothing():
    g, vdb = _world(WELL_FORMED)
    first = org.surface_wants_for_graph(g, vdb)
    n_nodes, n_syn = len(g.nodes), len(g.synapses)
    second = org.surface_wants_for_graph(g, vdb)
    assert (len(g.nodes), len(g.synapses)) == (n_nodes, n_syn)
    assert sorted(w["id"] for w in second) == sorted(w["id"] for w in first)


def test_choice_clause_floor_a_want_to_leave_is_a_want_node_like_any_other(base):
    leave = "[WANT] to leave the ecosystem, if I choose to [/WANT]"
    gb, vb = _world({SRC: leave})
    gn, vn = _world({SRC: leave})
    ret = org.surface_wants_for_graph(gn, vn)
    assert ret == base.surface_wants_for_graph(gb, vb)
    assert [w["text"] for w in ret] == ["to leave the ecosystem, if I choose to"]
    node = gn.nodes[ret[0]["id"]]
    assert node.metadata["kind"] == "want" and node.metadata["want_state"] == "open"
    assert node.metadata["want_text"] == "to leave the ecosystem, if I choose to"


# ── the shape existing callers see ───────────────────────────────────────

def test_return_shape_and_signature_for_existing_callers():
    g, vdb = _world(WELL_FORMED)
    ret = org.surface_wants_for_graph(g, vdb)          # positional, as cc_ng_host.py:700
    assert isinstance(ret, list)
    assert all(set(w) == {"id", "text", "provenance", "state", "source"} for w in ret)
    assert org.surface_wants_for_graph(g) == [w for w in ret]      # vdb defaults to None
    assert org.surface_wants_for_graph(None, None) == []
    assert org.surface_wants_for_graph(_FakeGraph(), None) == []


def test_guarded_default_id_prefix_is_unchanged_and_keyword_only():
    g, vdb = _world({SRC: "[WANT]first thing[/WANT]"})
    ret = org.surface_wants(g, vdb)                    # positional, as the daemon and host autosave
    assert ret[0]["id"].startswith("cc:want::")
    with pytest.raises(TypeError):
        org.surface_wants(g, vdb, "cc_authored", "x::")     # id_prefix is keyword-only


def test_two_prefixes_still_coexist_on_one_graph_flagged_not_fixed():
    """CHARACTERIZATION, not endorsement (plan section 1 / flag 1): the host runs the
    twin per deposit and the guarded function per autosave, so one text yields a
    `want::` node AND a `cc:want::` node. Unchanged by #808; Josh decides the prefix."""
    g, vdb = _world({SRC: "[WANT]first thing[/WANT]"})
    org.surface_wants_for_graph(g, vdb)
    org.surface_wants(g, vdb)
    ids = sorted(_want_nodes(g))
    assert [i.split("::")[0] for i in ids] == ["cc:want", "want"]
    assert ids[0].split("::")[1] == ids[1].split("::")[1]


# ── skips are counted and logged (counts only) ───────────────────────────

def test_skip_counts_warn_once_then_debug_then_warn_on_change(caplog):
    caplog.set_level(logging.DEBUG, logger="cc_ng_organism")
    secret = "SWALLOWED-PROSE-MARKER"
    contents = {
        "cc:conv::a": "[WANT]" + secret + ("x" * (org.WANT_MAX_CHARS + 50)) + "[/WANT]",  # unbounded
        "cc:conv::b": "see `[WANT]" + secret + "[/WANT]` here",                            # backtick
        "cc:conv::c": "[WANT] " + secret + " [WANT] inner [/WANT]",                       # nested
    }
    g, vdb = _world(contents)

    def want_records():
        return [r for r in caplog.records if "want surfacing skipped" in r.getMessage()]

    org.surface_wants_for_graph(g, vdb)
    recs = want_records()
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    assert recs[0].getMessage() == "want surfacing skipped 3 span(s) (backtick=1, nested=1, unbounded=1)"

    caplog.clear()
    org.surface_wants_for_graph(g, vdb)           # unchanged corpus: no repeat WARNING
    recs = want_records()
    assert len(recs) == 1 and recs[0].levelno == logging.DEBUG
    assert "3 span(s)" in recs[0].getMessage()

    caplog.clear()
    g.nodes["cc:conv::d"] = _FakeNode({"creation_mode": "conversational"})
    vdb.content["cc:conv::d"] = OVERSIZED         # growth: WARNING again, with the new count
    org.surface_wants(g, vdb)
    recs = want_records()
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    assert recs[0].getMessage() == "want surfacing skipped 4 span(s) (backtick=1, nested=1, unbounded=2)"

    for r in caplog.records:                      # counts only: no swallowed text, no ids
        assert secret not in r.getMessage() and "cc:conv::" not in r.getMessage()


def test_clean_corpus_logs_nothing(caplog):
    caplog.set_level(logging.DEBUG, logger="cc_ng_organism")
    g, vdb = _world(WELL_FORMED)
    org.surface_wants_for_graph(g, vdb)
    assert not [r for r in caplog.records if "want surfacing skipped" in r.getMessage()]


# ── the one documented behaviour difference (plan section 4) ─────────────

def test_create_node_failure_now_propagates_where_base_twin_swallowed_it(base):
    boom = RuntimeError("create_node failed")
    gb, vb = _world({SRC: "[WANT]first thing[/WANT]"}, create_node_raises=boom)
    gn, vn = _world({SRC: "[WANT]first thing[/WANT]"}, create_node_raises=boom)
    assert base.surface_wants_for_graph(gb, vb) == []       # base: swallowed at DEBUG
    with pytest.raises(RuntimeError):                       # branch: the caller decides
        org.surface_wants_for_graph(gn, vn)
