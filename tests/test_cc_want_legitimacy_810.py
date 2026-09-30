# ---- Changelog ----
# [2026-09-29] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810, Exec P406/P408)
# What: tests for the structural-legitimacy WANT parser (cc_ng_organism.parse_wants /
#   want_id_for_text / surface_wants): a whole 2,000-char want, the 2026-09-16 mis-parse shapes
#   rejected as mentions, nested / nearest-opener pairing, backticks inside a real pair,
#   fences (CRLF, tilde, unclosed, 4-space), idempotence, the flood-bounded INFO skip log
#   (reason, volume, no marker text), render_wants UNCHANGED vs base, a golden comparison
#   against BASE e4ebf982's surface_wants on well-formed input, and the P379 preamble.
# Why: Exec P406 (Josh): a WANT is the text between a real [WANT] and its paired [/WANT], no
#   length limit, legitimacy structural. Exec P408: render_wants is not touched this turn.
# How: fake in-memory graph + vdb only -- never a live path. The base module is loaded from
#   `git show e4ebf982:cc_ng_organism.py` under a private name; a missing base FAILS (never skips).
# -------------------
"""#810 -- the WANT parser is structural LEGITIMACY, not a length limit.

Run from the worktree root with NG_EMBED_* unset:
    env -u NG_EMBED_REMOTE python3 -m pytest tests/test_cc_want_legitimacy_810.py -s
"""
import ast
import hashlib
import inspect
import logging
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

import cc_ng_organism as org

_WORKTREE = Path(__file__).resolve().parents[1]
_BASE_COMMIT = "e4ebf982b1989fd9066d610b94853bc68bf70d37"
_NG_MODULES = ("cc_ng_organism", "neurograph_rpc", "neuro_foundation", "ng_lite", "ng_embed",
               "ng_ecosystem", "ng_tract_bridge", "ng_autonomic", "openclaw_adapter",
               "surface_resolver", "surfacing", "cc_ng_host")


# ---------------------------------------------------------------------------
# P379 preamble: which copy of the code is this session actually running?
# ---------------------------------------------------------------------------

def _p379_report():
    lines = []
    for name in _NG_MODULES:
        mod = sys.modules.get(name)
        path = getattr(mod, "__file__", None) if mod is not None else None
        lines.append("P379 %-18s %-14s %s" % (
            name, "in sys.modules" if mod is not None else "not loaded", path or "-"))
    embed_names = sorted(k for k in os.environ if k.startswith("NG_EMBED"))
    lines.append("P379 NG_EMBED_* names set in env: %s" % (embed_names or "none"))
    return lines


@pytest.fixture(scope="module", autouse=True)
def _p379_preamble():
    """Print resolved module paths at session start; FAIL every test here if cc_ng_organism
    (or any NG module already loaded) resolves outside this worktree."""
    print("\n" + "\n".join(_p379_report()))
    resolved = Path(org.__file__).resolve()
    assert resolved == _WORKTREE / "cc_ng_organism.py", (
        "cc_ng_organism is NOT the worktree copy: %s (expected %s)" % (
            resolved, _WORKTREE / "cc_ng_organism.py"))
    for name in _NG_MODULES:
        mod = sys.modules.get(name)
        path = getattr(mod, "__file__", None) if mod is not None else None
        if path:
            assert _WORKTREE in Path(path).resolve().parents, (
                "%s resolves outside the worktree: %s" % (name, path))
    yield


@pytest.fixture(autouse=True)
def _fresh_skip_log_state():
    org._reset_want_skip_log_state()
    yield
    org._reset_want_skip_log_state()


def test_p379_preamble_resolves_worktree_copy():
    # the module-scoped autouse fixture above already FAILS the whole file otherwise; this
    # is the visible line item
    assert Path(org.__file__).resolve() == _WORKTREE / "cc_ng_organism.py"


# ---------------------------------------------------------------------------
# fakes (in-memory only)
# ---------------------------------------------------------------------------

class _FakeNode:
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})
        self.creation_time = 0.0


class _FakeGraph:
    def __init__(self):
        self.nodes = {}
        self.synapses = []
        self._step_lock = threading.RLock()

    def create_node(self, node_id, metadata=None):
        n = _FakeNode(metadata)
        self.nodes[node_id] = n
        return n

    def create_synapse(self, a, b, weight=0.0):
        self.synapses.append((a, b, weight))


class _FakeVDB:
    def __init__(self, content):
        self.content = content


def _graph_with(*contents):
    g = _FakeGraph()
    vdb = {}
    for i, text in enumerate(contents):
        nid = "cc:conv::src%d" % i
        g.nodes[nid] = _FakeNode({"creation_mode": "conversational"})
        vdb[nid] = text
    return g, _FakeVDB(vdb)


def _texts(wants):
    return [w["text"] for w in wants]


def _want_nodes(g):
    return {nid: n.metadata for nid, n in g.nodes.items() if n.metadata.get("kind") == "want"}


def _sha(text):
    return "cc:want::" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# no length limit
# ---------------------------------------------------------------------------

def test_real_2000_char_want_is_captured_whole():
    body = ("I want to understand why the recall path drops the long ones. " * 40)[:2000].strip()
    assert len(body) >= 1990
    g, vdb = _graph_with("before [WANT]" + body + "[/WANT] after")
    wants = org.surface_wants(g, vdb)
    assert _texts(wants) == [body]
    nodes = _want_nodes(g)
    assert list(nodes) == [_sha(body)]
    assert nodes[_sha(body)]["want_text"] == body            # stored whole, not clipped
    assert len(nodes[_sha(body)]["want_text"]) > org.WANT_MAX_CHARS


def test_very_long_want_has_no_upper_bound():
    body = "z" * 50_000
    parsed = org.parse_wants("[WANT]" + body + "[/WANT]")
    assert [w.text for w in parsed.wants] == [body]
    assert parsed.skipped == ()


def test_want_max_chars_is_no_longer_in_the_parser():
    src = inspect.getsource(org.parse_wants) + inspect.getsource(org.surface_wants)
    assert "WANT_MAX_CHARS" not in src
    assert not hasattr(org, "_WANT_RE")              # the 600-char pattern is gone


# ---------------------------------------------------------------------------
# the 2026-09-16 mis-parse shapes are MENTIONS
# ---------------------------------------------------------------------------

_FILLER = "filler discussion of the architecture. " * 60

_MENTION_SHAPES = {
    "backticked mention running to a far closer": (
        "The loop dispatches on markers:\n- `[WANT]` -> write to the wants register\n\n"
        + _FILLER + "\n\nand the closing `[/WANT]` ends the span.\n"),
    "backtick-led opener, bare far closer": (
        "use `[WANT]` -> register it.\n\n" + _FILLER + "\n\nlater a bare [/WANT] appears"),
    "fenced block": "```\n[WANT] documented in a fence [/WANT]\n```\n",
    "fenced block, tilde": "~~~text\n[WANT] documented [/WANT]\n~~~\n",
    "fenced block, CRLF": "```\r\n[WANT] documented [/WANT]\r\n```\r\n",
    "unclosed fence runs to the end": "intro\n```\n[WANT] documented [/WANT]\nmore\n",
    "quoted mention": 'the "[WANT]" tag opens and the "[/WANT]" tag closes it',
    "curly-quoted mention": "the “[WANT]” tag and the “[/WANT]” tag",
    "escaped mention": r"write \[WANT] then \[/WANT] literally",
    "backtick right before the opener": "see foo`[WANT] something [/WANT]",
}


@pytest.mark.parametrize("name", sorted(_MENTION_SHAPES))
def test_mention_shape_creates_no_want(name):
    g, vdb = _graph_with(_MENTION_SHAPES[name])
    assert org.surface_wants(g, vdb) == []
    assert _want_nodes(g) == {}
    skipped = org.parse_wants(_MENTION_SHAPES[name]).skipped
    assert skipped, "a mention must be reported as skipped, never silently ignored"
    assert {s.reason for s in skipped} <= set(org.WANT_SKIP_REASONS)


def test_mention_reasons_are_the_specific_ones():
    def reasons(text):
        return {s.reason for s in org.parse_wants(text).skipped}
    assert reasons(_MENTION_SHAPES["backticked mention running to a far closer"]) == {"in_code_span"}
    assert reasons(_MENTION_SHAPES["fenced block"]) == {"in_fence"}
    assert reasons(_MENTION_SHAPES["quoted mention"]) == {"quoted"}
    assert reasons(_MENTION_SHAPES["escaped mention"]) == {"escaped"}
    assert "code_adjacent" in reasons(_MENTION_SHAPES["backtick right before the opener"])


def test_real_want_after_a_mention_is_recovered():
    content = "the `[WANT]` marker opens one.\n\n" + _FILLER + "\n\n[WANT] a real forward intent [/WANT]"
    g, vdb = _graph_with(content)
    assert _texts(org.surface_wants(g, vdb)) == ["a real forward intent"]


def test_double_backslash_is_not_an_escape():
    assert [w.text for w in org.parse_wants(r"\\[WANT] real [/WANT]").wants] == ["real"]


def test_quote_must_hug_the_marker_token():
    parsed = org.parse_wants('He wrote "[WANT]fix x[/WANT]" in the note')
    assert [w.text for w in parsed.wants] == ["fix x"]
    assert parsed.skipped == ()


def test_fence_edge_cases():
    # 4-space indent is NOT a fence (indented code blocks are not recognised)
    assert [w.text for w in org.parse_wants("    ```\n[WANT] kept [/WANT]\n").wants] == ["kept"]
    # ```inline``` on one line is a code span, not a fence; the real want after it survives
    parsed = org.parse_wants("x ```[WANT]``` y\n[WANT] kept [/WANT]")
    assert [w.text for w in parsed.wants] == ["kept"]
    assert [s.reason for s in parsed.skipped] == ["in_code_span"]
    # a longer closing fence closes; a shorter one does not
    parsed = org.parse_wants("````\n[WANT] a [/WANT]\n```\n[WANT] b [/WANT]\n````\n[WANT] c [/WANT]")
    assert [w.text for w in parsed.wants] == ["c"]


def test_unbalanced_backtick_is_literal_and_masks_nothing():
    assert [w.text for w in org.parse_wants("it's a ` character. [WANT] real [/WANT]").wants] == ["real"]
    assert [w.text for w in org.parse_wants("a ` b\n\nc ` [WANT] real [/WANT]").wants] == ["real"]


# ---------------------------------------------------------------------------
# pairing: nested, nearest opener, stray markers
# ---------------------------------------------------------------------------

def test_nested_opener_is_rejected_and_closer_pairs_with_nearest_opener():
    parsed = org.parse_wants("[WANT] outer [WANT] inner [/WANT]")
    assert [w.text for w in parsed.wants] == ["inner"]          # never "outer [WANT] inner"
    assert [(s.marker, s.start, s.reason) for s in parsed.skipped] == [("[WANT]", 0, "opener_unclosed")]


def test_nested_with_an_extra_closer():
    parsed = org.parse_wants("[WANT] a [WANT] b [/WANT] c [/WANT]")
    assert [w.text for w in parsed.wants] == ["b"]
    assert sorted(s.reason for s in parsed.skipped) == ["closer_without_opener", "opener_unclosed"]


def test_a_returned_want_never_contains_a_live_marker():
    for content in ("[WANT] a [WANT] b [/WANT]", "[WANT] a [/WANT] [/WANT] [WANT] c [/WANT]",
                    "[/WANT][WANT]x[/WANT][WANT]", "[WANT][WANT][WANT] d [/WANT]"):
        for w in org.parse_wants(content).wants:
            assert "[WANT]" not in w.text and "[/WANT]" not in w.text


def test_stray_closer_before_any_opener():
    parsed = org.parse_wants("[/WANT] stray [WANT] ok [/WANT]")
    assert [w.text for w in parsed.wants] == ["ok"]
    assert [(s.marker, s.reason) for s in parsed.skipped] == [("[/WANT]", "closer_without_opener")]


def test_unclosed_opener_and_empty_pair_are_skipped_not_silent():
    assert org.parse_wants("[WANT] never closed").wants == ()
    assert [s.reason for s in org.parse_wants("[WANT] never closed").skipped] == ["opener_unclosed"]
    parsed = org.parse_wants("[WANT]   [/WANT] [WANT]z[/WANT]")
    assert [w.text for w in parsed.wants] == ["z"]
    assert [s.reason for s in parsed.skipped] == ["empty_pair", "empty_pair"]


def test_adjacent_pairs_both_captured_in_source_order():
    parsed = org.parse_wants("[WANT]a[/WANT][WANT]b[/WANT] and [WANT]c[/WANT]")
    assert [w.text for w in parsed.wants] == ["a", "b", "c"]
    assert [(w.open_start, w.close_end) for w in parsed.wants] == [(0, 14), (14, 28), (33, 47)]


# ---------------------------------------------------------------------------
# backticks INSIDE a real pair
# ---------------------------------------------------------------------------

def test_want_containing_backticks_inside_a_real_pair_is_captured():
    body = "revisit how `[WANT]` renders and whether `parse_wants()` keeps `code` intact"
    g, vdb = _graph_with("[WANT] " + body + " [/WANT]")
    assert _texts(org.surface_wants(g, vdb)) == [body]          # the backticked marker is text, not a marker
    parsed = org.parse_wants("[WANT] " + body + " [/WANT]")
    assert [s.reason for s in parsed.skipped] == ["in_code_span"]   # ...and is still reported


def test_want_that_begins_with_inline_code_is_a_real_want():
    parsed = org.parse_wants("[WANT]`cc_ng_host` needs a follow-up[/WANT]")
    assert [w.text for w in parsed.wants] == ["`cc_ng_host` needs a follow-up"]
    assert parsed.skipped == ()


# ---------------------------------------------------------------------------
# text, ids, idempotence
# ---------------------------------------------------------------------------

def test_text_is_stripped_only_and_ids_use_the_one_function():
    inner = "  line one\r\nline two é中  "
    parsed = org.parse_wants("[WANT]" + inner + "[/WANT]")
    want = parsed.wants[0]
    assert want.text == inner.strip()                          # CRLF and unicode untouched inside
    assert want.want_id == org.want_id_for_text(want.text) == _sha(want.text)
    g, vdb = _graph_with("[WANT]" + inner + "[/WANT]")
    org.surface_wants(g, vdb)
    assert list(_want_nodes(g)) == [want.want_id]


def test_surface_wants_is_idempotent():
    g, vdb = _graph_with("[WANT] first [/WANT] `[WANT]` mention [WANT] second [/WANT]",
                         "[WANT]" + "q" * 900 + "[/WANT]")
    first = org.surface_wants(g, vdb)
    nodes_after_first = {nid: dict(m) for nid, m in _want_nodes(g).items()}
    synapses_after_first = list(g.synapses)
    second = org.surface_wants(g, vdb)
    assert sorted(w["id"] for w in second) == sorted(w["id"] for w in first)
    assert {nid: dict(m) for nid, m in _want_nodes(g).items()} == nodes_after_first
    assert g.synapses == synapses_after_first                  # no duplicate synapses either
    assert org.parse_wants(vdb.content["cc:conv::src0"]) == org.parse_wants(vdb.content["cc:conv::src0"])


# ---------------------------------------------------------------------------
# flood-safe INFO skip log
# ---------------------------------------------------------------------------

_LOGGER = "cc_ng_organism"


def _info(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == _LOGGER and r.levelno == logging.INFO and "surface_wants: skipped" in r.getMessage()]


class _Clock:
    """Stand-in for org.time: only monotonic() is overridden, nothing global is patched."""
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        import time as _time
        return getattr(_time, name)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(org, "time", c)
    return c


def test_skip_is_logged_at_info_with_reason_node_and_offset(caplog, clock):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    g, vdb = _graph_with("SECRETSENTINEL-prefix `[WANT]` SECRETSENTINEL-mid [/WANT] SECRETSENTINEL-end")
    org.surface_wants(g, vdb)
    lines = _info(caplog)
    summary = [l for l in lines if "marker(s) in" in l]
    detail = [l for l in lines if "node=" in l]
    assert len(summary) == 1 and "in_code_span=1" in summary[0] and "closer_without_opener=1" in summary[0]
    assert "2 marker(s) in 1 node(s)" in summary[0]
    assert any("node=cc:conv::src0" in l and "reason=in_code_span" in l and "marker=" not in l
               and "[WANT]" in l for l in detail)
    assert any("reason=closer_without_opener" in l for l in detail)
    # never any marker/surrounding TEXT in the log
    assert all("SECRETSENTINEL" not in r.getMessage() for r in caplog.records)
    assert all("surface_wants" not in l or "`" not in l for l in lines)


def test_repeated_pulses_over_an_unchanged_corpus_are_quiet(caplog, clock):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    g, vdb = _graph_with("the `[WANT]` mention [/WANT]", "```\n[WANT] x [/WANT]\n```")
    org.surface_wants(g, vdb)
    first_call = len(_info(caplog))
    assert first_call >= 2                                     # a summary + details
    for _ in range(30):                                        # 30 autosave pulses (~30 min)
        clock.now += 60
        org.surface_wants(g, vdb)
    assert len(_info(caplog)) == first_call                    # nothing new within the hour
    clock.now += org.WANT_SKIP_SUMMARY_INTERVAL_S
    org.surface_wants(g, vdb)
    lines = _info(caplog)
    assert len(lines) == first_call + 1                        # exactly one heartbeat summary
    assert "marker(s) in" in lines[-1]


def test_a_new_mention_node_is_reported_promptly(caplog, clock):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    g, vdb = _graph_with("the `[WANT]` mention [/WANT]")
    org.surface_wants(g, vdb)
    before = len(_info(caplog))
    g.nodes["cc:conv::late"] = _FakeNode({"creation_mode": "conversational"})
    vdb.content["cc:conv::late"] = 'the "[WANT]" tag'
    clock.now += 60
    org.surface_wants(g, vdb)
    new = _info(caplog)[before:]
    assert any("marker(s) in 2 node(s)" in l for l in new)     # counts changed -> new summary
    assert any("node=cc:conv::late" in l and "reason=quoted" in l for l in new)


def test_detail_lines_per_call_are_capped_and_drain_over_later_pulses(caplog, clock):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    n = org.WANT_SKIP_DETAIL_PER_CALL_MAX * 2 + 20
    g, vdb = _graph_with(*["stray [/WANT] %d" % i for i in range(n)])
    per_call = []
    for _ in range(5):
        before = len(_info(caplog))
        clock.now += 60
        org.surface_wants(g, vdb)
        per_call.append(len(_info(caplog)) - before)
    assert max(per_call) <= org.WANT_SKIP_DETAIL_PER_CALL_MAX + 1      # details + at most one summary
    assert per_call[0] == org.WANT_SKIP_DETAIL_PER_CALL_MAX + 1
    assert per_call[-1] == 0                                           # fully drained, then silent
    details = [l for l in _info(caplog) if "node=" in l]
    assert len(details) == n and len(set(details)) == n                # each marker detailed exactly once
    assert any("deferred to later pulses" in l for l in _info(caplog))


def test_seen_set_is_a_bounded_fifo(monkeypatch, clock):
    monkeypatch.setattr(org, "WANT_SKIP_SEEN_MAX", 10)
    g, vdb = _graph_with(*["stray [/WANT] %d" % i for i in range(30)])
    for _ in range(3):
        org.surface_wants(g, vdb)
    assert len(org._WANT_SKIP_SEEN) <= 10


def test_clean_corpus_logs_nothing(caplog, clock):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    g, vdb = _graph_with("[WANT] a clean want [/WANT]", "no markers at all")
    org.surface_wants(g, vdb)
    assert _info(caplog) == []


def test_logging_happens_after_the_graph_lock_is_released(monkeypatch, clock):
    g, vdb = _graph_with("stray [/WANT]")
    held = []

    def probe(events):
        held.append(g._step_lock._is_owned())
    monkeypatch.setattr(org, "_log_want_skips", probe)
    org.surface_wants(g, vdb)
    assert held == [False]


# ---------------------------------------------------------------------------
# render_wants is NOT touched (Exec P408: it is retired in a separate turn)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_org():
    """BASE e4ebf982's cc_ng_organism.py, loaded under a private module name."""
    proc = subprocess.run(["git", "-C", str(_WORKTREE), "show", "%s:cc_ng_organism.py" % _BASE_COMMIT],
                          capture_output=True)
    if proc.returncode != 0:
        pytest.fail("cannot read base %s via git show (golden test must not be skipped): %s"
                    % (_BASE_COMMIT, proc.stderr.decode(errors="replace")[:300]))
    name = "cc_ng_organism_base_e4ebf982"
    mod = types.ModuleType(name)
    mod.__file__ = str(_WORKTREE / "cc_ng_organism.py")
    mod._base_source = proc.stdout.decode("utf-8")
    sys.modules[name] = mod
    try:
        exec(compile(mod._base_source, "<base e4ebf982 cc_ng_organism.py>", "exec"), mod.__dict__)
        yield mod
    finally:
        sys.modules.pop(name, None)


def _function_source(source_text, name):
    """Exact source segment of top-level function `name` in a module's source text."""
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source_text, node)
    raise AssertionError("function %s not found" % name)


def test_render_wants_source_is_byte_identical_to_base(base_org):
    current = (_WORKTREE / "cc_ng_organism.py").read_text(encoding="utf-8")
    assert _function_source(current, "render_wants") == _function_source(base_org._base_source, "render_wants")
    assert org.WANT_RENDER_LIMIT == base_org.WANT_RENDER_LIMIT == 40
    assert org.WANT_MAX_CHARS == base_org.WANT_MAX_CHARS == 600      # still defined: the renderer uses it


def _render_graph(texts):
    g = _FakeGraph()
    for i, t in enumerate(texts):
        n = _FakeNode({"kind": "want", "want_text": t, "want_state": "open", "provenance": "cc_authored"})
        n.creation_time = float(i)
        g.nodes["cc:want::%03d" % i] = n
    return g


def test_render_wants_output_equals_base_including_the_held_clamp(base_org):
    for texts in (["a small want"], ["w" * 1500, "short"], ["x%03d " % i + "y" * 700 for i in range(55)], []):
        assert org.render_wants(_render_graph(texts)) == base_org.render_wants(_render_graph(texts))
    # the held behaviour, stated plainly: a >600 want is still clipped by the (untouched) renderer
    assert org.render_wants(_render_graph(["w" * 1500])) == "## What I Want\n- " + "w" * 600


# ---------------------------------------------------------------------------
# golden comparison against BASE e4ebf982 on well-formed input
# ---------------------------------------------------------------------------

def _run(mod, contents, extra_nodes=()):
    g, vdb = _graph_with(*contents)
    for nid, meta in extra_nodes:
        g.nodes[nid] = _FakeNode(meta)
    returned = mod.surface_wants(g, vdb)
    nodes = {nid: dict(n.metadata) for nid, n in g.nodes.items()}
    return returned, nodes, list(g.synapses)


_GOLDEN_CORPUS = {
    "single short want": ["thinking out loud [WANT] follow up on the numpy/scipy conflict later [/WANT] anyway"],
    "two wants one node": ["[WANT] first thing [/WANT] noise [WANT] second thing [/WANT]"],
    "want at very start and very end": ["[WANT]a[/WANT]", "x [WANT]tail want[/WANT]"],
    "padded whitespace and newlines": ["[WANT]\n  padded\n  multi-line want\n[/WANT]"],
    "unicode": ["[WANT] revisit éè 中文 handling [/WANT]"],
    "CRLF inside": ["[WANT]line one\r\nline two[/WANT]"],
    "duplicate text twice and across nodes": ["[WANT] same [/WANT] [WANT] same [/WANT]", "[WANT] same [/WANT]"],
    "inline code inside, no markers": ["[WANT] make `parse()` and `emit()` agree [/WANT]"],
    "exactly 600 chars": ["[WANT]" + "y" * 600 + "[/WANT]"],
    "no markers, other brackets": ["[WANTED] poster [want] lower [ WANT ] spaced"],
    "many nodes": ["[WANT] n%d [/WANT]" % i for i in range(25)],
}


@pytest.mark.parametrize("name", sorted(_GOLDEN_CORPUS))
def test_golden_same_id_node_metadata_and_synapse_as_base(base_org, name):
    new = _run(org, _GOLDEN_CORPUS[name])
    old = _run(base_org, _GOLDEN_CORPUS[name])
    assert new == old
    assert new[0] or name == "no markers, other brackets"      # the comparison is not vacuous


def test_golden_with_preexisting_want_nodes_and_non_conversational_sources(base_org):
    extra = [
        ("cc:want::old_open", {"kind": "want", "want_text": "kept", "want_state": "open",
                               "provenance": "cc_emergent", "source_node": None}),
        ("cc:want::old_done", {"kind": "want", "want_text": "done", "want_state": "closed"}),
        ("cc:conv::ingested", {"creation_mode": "ingested"}),
    ]
    contents = ["[WANT] real one [/WANT]"]
    assert _run(org, contents, extra) == _run(base_org, contents, extra)


def test_golden_returned_want_dict_shape_and_provenance_argument(base_org):
    g1, v1 = _graph_with("[WANT] p [/WANT]")
    g2, v2 = _graph_with("[WANT] p [/WANT]")
    assert org.surface_wants(g1, v1, provenance="cc_test") == base_org.surface_wants(g2, v2, provenance="cc_test")
    assert org.surface_wants(None, None) == base_org.surface_wants(None, None) == []


# Inputs where the rule DELIBERATELY changed: (content, base result, new result, reason)
_DELTAS = {
    "over the old 600 cap": ("[WANT]" + "y" * 601 + "[/WANT]", [], ["y" * 601],
                             "no length limit (P406)"),
    "nested: nearest opener pairs": ("[WANT] outer [WANT] inner [/WANT]", [], ["inner"],
                                     "closer pairs with the NEAREST opener"),
    "fenced mention": ("```\n[WANT] documented [/WANT]\n```", ["documented"], [],
                       "fenced block is a mention"),
    "quoted mention": ('the "[WANT]" tag and "[/WANT]" end', ['" tag and "'], [],
                       "quote-wrapped token is a mention"),
    "backticked marker inside a real pair": ("[WANT] see `[WANT]` docs [/WANT]", [], ["see `[WANT]` docs"],
                                            "a masked marker is text, not a marker"),
}


@pytest.mark.parametrize("name", sorted(_DELTAS))
def test_documented_deltas_against_base(base_org, name):
    content, base_expected, new_expected, _reason = _DELTAS[name]
    assert _texts(_run(base_org, [content])[0]) == base_expected
    assert _texts(_run(org, [content])[0]) == new_expected
