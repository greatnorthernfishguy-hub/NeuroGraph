# ---- Changelog ----
# [2026-10-04] Claude (lane 810-onto-s4) — rebased onto trial s4 (040be4d): render_wants is compared to the trial base
#   (_RENDER_BASE_COMMIT; the trial added an on_error reporter to it, #810 still leaves it untouched); the two
#   changelog-claim tests read the WHOLE header (the trial header grew past the old 40,000-char slice).
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810 turn 2, #815)
# What: tests for in_json_string / in_url / in_link_target: checker-016's exact examples (incl.
#   a backtick-fence variant and its independence from the inline-code coincidence) rejected with
#   their reasons; a genuine >600-char want in plain prose beside each (4 arrangements) captured
#   whole; the boundary (reason applies to the OPENER's context, not the want text); a pinned
#   list of what is NOT recognised; log flow unchanged; linearity on adversarial input; the
#   parity corpus widened to 3,726 shapes with URL/link/JSON/escaped-quote near-misses.
# Why: Exec P414. Precision over cleverness: each rule's failure modes are pinned.
# How: same fakes; no graph load.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810 turn 2, le-014)
# What: golden cases ASSERTED AGAINST BASE e4ebf982 for a want that ENDS in inline code, BEGINS
#   with it, both, with a follow-on second want (C1); a closer wrapped in quotes (C2); a want
#   ending in a backslash / a URL; a 1,980-shape combinatorial parity corpus; an adversarial
#   'real want next to code/quotes/stray backticks' corpus; the stray-backtick residual pinned
#   as documented; env-sourcing of the three WANT_SKIP_* bounds; the exact flood claim.
#   Two existing reason assertions updated (an orphan closer is closer_without_opener).
# Why: le-014 C1 (HIGH) found a base regression the turn-1 corpus never exercised (it put
#   backticks only mid-want). The #801 id-equality guarantee is a property of SHAPES.
# How: same fakes; base loaded via `git show`; a missing base FAILS, never skips.
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
import itertools
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
# Trial base #810 was rebased onto (lane 810-onto-s4): render_wants is compared to THIS, because the trial
# itself changed render_wants (on_error reporter) after e4ebf982; #810 still must not touch it.
_RENDER_BASE_COMMIT = "040be4df68b6b4924d0a598cfaaf967119fce708"
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
    # adjacency guesses (quote / escape / backtick) are OPENER-only (le-014 C1/C2): the orphan
    # closer of a mentioned opener is skipped as a stray closer, not as the same guess
    assert reasons(_MENTION_SHAPES["quoted mention"]) == {"quoted", "closer_without_opener"}
    assert reasons(_MENTION_SHAPES["escaped mention"]) == {"escaped", "closer_without_opener"}
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
    proc = subprocess.run(["git", "-C", str(_WORKTREE), "show", "%s:cc_ng_organism.py" % _RENDER_BASE_COMMIT],
                          capture_output=True)
    if proc.returncode != 0:
        pytest.fail("cannot read render base %s via git show (must not be skipped): %s"
                    % (_RENDER_BASE_COMMIT, proc.stderr.decode(errors="replace")[:300]))
    assert _function_source(current, "render_wants") == _function_source(proc.stdout.decode("utf-8"), "render_wants")
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


# ---------------------------------------------------------------------------
# le-014 corrections (turn 2): the parser equals base for every well-formed want shape
# ---------------------------------------------------------------------------

# C1/C2 and their neighbours. Every one is asserted against BASE and against an explicit list,
# so a future guard cannot silently suppress a real want that base minted.
_LE014_GOLDEN = {
    "C1 ends in inline code": (
        "I noticed it. [WANT]check `foo()`[/WANT] done.", ["check `foo()`"]),
    "C1 follow-on want after one ending in code": (
        "[WANT]fix `a`[/WANT] and later [WANT]rest[/WANT]", ["fix `a`", "rest"]),
    "C1 second follow-on shape": (
        "[WANT]do `x`[/WANT] tail [WANT]second[/WANT]", ["do `x`", "second"]),
    "begins with inline code": (
        "[WANT]`foo()` needs work[/WANT]", ["`foo()` needs work"]),
    "begins and ends with inline code": (
        "[WANT]`foo()` and `bar()`[/WANT] ok", ["`foo()` and `bar()`"]),
    "C2 closer wrapped in a quote pair": (
        'x "[WANT]I want "x"[/WANT]" y', ['I want "x"']),
    "ends in a backslash": (
        "[WANT]path is C:\\[/WANT] ok", ["path is C:\\"]),
    "ends in a URL": (
        "[WANT]read https://example.com/docs[/WANT] then", ["read https://example.com/docs"]),
    "want ending in a URL, then an adjacent want": (
        "[WANT]read https://x.com/a[/WANT][WANT]second[/WANT]", ["read https://x.com/a", "second"]),
    "want that contains a URL, a quoted word and code": (
        'revisit [WANT]see "docs" at https://x.org/a and `code`[/WANT] thanks',
        ['see "docs" at https://x.org/a and `code`']),
    "opener right after a closing backtick (guard kept, same as base)": (
        "see `x`[WANT]real[/WANT]", []),
}


@pytest.mark.parametrize("name", sorted(_LE014_GOLDEN))
def test_le014_golden_against_base(base_org, name):
    content, expected = _LE014_GOLDEN[name]
    assert _texts(_run(base_org, [content])[0]) == expected, "the expectation must be what BASE mints"
    assert _run(org, [content]) == _run(base_org, [content])
    assert _texts(_run(org, [content])[0]) == expected


def test_closer_adjacency_guesses_do_not_apply_to_closers():
    # C1: a closer right after a backtick is a real closer
    parsed = org.parse_wants("[WANT]check `foo()`[/WANT]")
    assert [w.text for w in parsed.wants] == ["check `foo()`"] and parsed.skipped == ()
    # C2: a closer wrapped in a quote pair is a real closer
    parsed = org.parse_wants('[WANT]I want "x"[/WANT]" y')
    assert [w.text for w in parsed.wants] == ['I want "x"']
    # a closer after an odd backslash is a real closer (a want may end in a path separator)
    parsed = org.parse_wants("[WANT]path is C:\\[/WANT]")
    assert [w.text for w in parsed.wants] == ["path is C:\\"]
    # ...while the OPENER guesses still hold
    assert org.parse_wants("x`[WANT]a[/WANT]").wants == ()
    assert org.parse_wants('"[WANT]" a "[/WANT]"').wants == ()
    assert org.parse_wants("\\[WANT]a[/WANT]").wants == ()


def test_structural_masks_judge_a_closer_relative_to_its_opener():
    """turn 3 (le-016 #3): a closer in a fence / code span with a REAL opener pending and no mention
    opener in that region is the real want's closer (base parity). A mention PAIR inside one
    region is still masked, so a want may discuss markup."""
    parsed = org.parse_wants("[WANT] replace `[/WANT]` tokens [/WANT]")
    assert [w.text for w in parsed.wants] == ["replace `"]                      # = base
    assert [(s.marker, s.reason) for s in parsed.skipped] == [("[/WANT]", "closer_without_opener")]
    parsed = org.parse_wants("[WANT] a\n```\n[/WANT]\n```\n b [/WANT]")
    assert [w.text for w in parsed.wants] == ["a\n```"]                        # = base
    # a mention pair inside ONE region stays masked: the want keeps it as text
    for content, text in (
            ("[WANT] see\n```\n[WANT]x[/WANT]\n```\n ok [/WANT]", "see\n```\n[WANT]x[/WANT]\n```\n ok"),
            ("[WANT]see `[WANT]z[/WANT]` ok[/WANT]", "see `[WANT]z[/WANT]` ok"),
            ('[WANT]use {"k": "[WANT]z[/WANT]"} form[/WANT]', 'use {"k": "[WANT]z[/WANT]"} form')):
        parsed = org.parse_wants(content)
        assert [w.text for w in parsed.wants] == [text], content
        assert sorted(s.marker for s in parsed.skipped) == ["[/WANT]", "[WANT]"]


_PARITY_PREFIXES = ["", "Noted. ", "Line one.\n", "para\n\n", "see `x` and ", 'he said "hi" then ',
                    "path C:\\dir ", "values: 1, 2, ", "a (b) ", "Ok!\r\n", "emoji \u2728 ",
                    # #815 near-misses: URL / link / JSON-ish / escaped quotes NEAR a real opener
                    "see https://example.com/docs ", "(https://example.com/docs)", "see https://example.com/docs.",
                    'he typed \\"go\\" then ', '{"a": "b"} and ', "[link](https://x.org/a) then ",
                    "[link](https://x.org/a)",
                    # turn 3: glue characters between a URL and a real opener, and index/call shapes
                    "See https://x.org/a:", "See https://x.org/a\u2014", "See **https://x.org/a**",
                    "_https://x.org/a_", "https://x.org/a\\n", "See https://x.org/a~", "arr[0](./d/"]
_PARITY_BODIES = ["plain want", "`code` begins", "ends `code()`", "`a` both `b`", "mid `x` code here",
                  'has "quoted" word', 'ends with "quote"', '"begins with quote" here',
                  "ends with backslash \\", "visit https://example.com/docs now",
                  "read https://example.com/docs", '{"a": "b"} json inside', "multi\nline\nbody",
                  "unicode \u00e9\u4e2d", "crlf\r\nbody", "ends with url https://x.org/a?b=[1]",
                  "with (parens) and [brackets]", "tab\tinside", "pair ``double`` end", 'x "[y" z',
                  "see [the docs](https://example.com/x)", '["a", "b"] list', "ends with link [d](https://x.org)",
                  # turn 3: want text that ENDS inside a quote fragment / JSON-looking literal
                  'rename "a", "b', 'list 1, "b', 'use {"a": "b', "stray ` tick",
                  "ends with a fence\n```\ncode\n```\n"]
_PARITY_SUFFIXES = ["", " tail", "\ntail", " [WANT]second[/WANT]", "[WANT]adj[/WANT]",
                    " and `code` after", "\n\n```\nfence after\n```\n", ' and "q" after', " `",
                    # turn 3: a closer followed by JSON-ish closers
                    '", next', '"}', '",\n']


def test_parity_with_base_on_every_well_formed_shape(base_org):
    """The property the #801 id-equality guarantee rests on: for EVERY combination of
    {prose prefix} x {want body, incl. begins/ends in code, quotes, backslash, URL, JSON-ish}
    x {suffix, incl. a follow-on want} the parser returns base's wants, ids, node metadata and
    synapses. No marker/fence/quote structure wraps the real opener in any of them; shapes
    that do are the documented deltas and the adversarial list below."""
    total = 0
    diverged = []
    for prefix, body, suffix in itertools.product(_PARITY_PREFIXES, _PARITY_BODIES, _PARITY_SUFFIXES):
        content = prefix + "[WANT]" + body + "[/WANT]" + suffix
        total += 1
        if _run(org, [content]) != _run(base_org, [content]):
            diverged.append(content)
    assert total == len(_PARITY_PREFIXES) * len(_PARITY_BODIES) * len(_PARITY_SUFFIXES) >= 1900
    assert diverged == [], "diverged from base on %d/%d shapes, first: %r" % (len(diverged), total, diverged[:3])


# 'a real want next to code / quotes / stray backticks' -- asserted against base so that no
# future guard can silently suppress one of these.
_ADVERSARIAL_PARITY = [
    "see `x`, then [WANT]real want[/WANT] and `y`",
    'he said "go" [WANT]real want[/WANT] "later"',
    "[WANT]a[/WANT] `b` [WANT]c[/WANT] `d`",
    "`a` [WANT]real[/WANT]",
    "``double`` [WANT]real with `inner` code[/WANT] ``double``",
    "'[WANT]real[/WANT]'",
    "(see [WANT]real want[/WANT])",
    "**bold** [WANT]real want[/WANT] *em*",
    "it's a ` character.\n\n[WANT]real[/WANT]\n\nanother ` char",
    "[WANT]unbalanced quote \" here[/WANT] and [WANT]second[/WANT]",
    "[WANT]ends with a quote char '[/WANT]",
    # #815: a real want next to a URL / link / JSON / escaped quotes is still a want
    "see https://example.com/docs [WANT]real want[/WANT]",
    "(https://x.org/a)[WANT]real want[/WANT]",
    "typed \\\"go\\\" [WANT]real want[/WANT]",
    '{"a": "b"} then [WANT]real want[/WANT]',
    "[WANT]real want ending in a link [d](https://x.org)[/WANT]",
    # turn 3: a stray backtick INSIDE a real want no longer masks its closer (le-014 C3 second shape)
    "[WANT]fix `a[/WANT] and `b` here",
]


@pytest.mark.parametrize("content", _ADVERSARIAL_PARITY)
def test_adversarial_real_want_near_code_quotes_backticks_equals_base(base_org, content):
    base_result = _run(base_org, [content])
    assert base_result[0], "corpus error: base must mint something here"
    assert _run(org, [content]) == base_result


# Known, documented divergences that are NOT deliberate features but residuals (le-014 C3):
# CommonMark-faithful, a real-want false negative whose only trace is an INFO skip line.
_ADVERSARIAL_RESIDUAL = {
    "stray backtick masks a real pair in the same paragraph": (
        "I saw `foo. Then [WANT]do X[/WANT] later `bar` end", ["do X"], []),
}


@pytest.mark.parametrize("name", sorted(_ADVERSARIAL_RESIDUAL))
def test_stray_backtick_residual_is_exactly_as_documented(base_org, name):
    content, base_expected, new_expected = _ADVERSARIAL_RESIDUAL[name]
    assert _texts(_run(base_org, [content])[0]) == base_expected
    assert _texts(_run(org, [content])[0]) == new_expected
    skipped = org.parse_wants(content).skipped
    assert skipped and {s.reason for s in skipped} <= {"in_code_span", "opener_unclosed"}   # visible, never silent


# ---------------------------------------------------------------------------
# LAW 5: the log bounds come from the environment; the flood claim is exact
# ---------------------------------------------------------------------------

def _import_with_env(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NG_EMBED", "CC_WANT_SKIP"))}
    env.update(env_extra)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out = subprocess.run(
        [sys.executable, "-c",
         "import cc_ng_organism as o; print(o.WANT_SKIP_SUMMARY_INTERVAL_S, o.WANT_SKIP_SEEN_MAX, "
         "o.WANT_SKIP_DETAIL_PER_CALL_MAX)"],
        cwd=str(_WORKTREE), env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    return out.stdout.split()


def test_skip_log_bounds_are_env_sourced_with_current_values_as_defaults():
    assert _import_with_env({}) == ["3600", "4096", "50"]
    assert _import_with_env({"CC_WANT_SKIP_SUMMARY_INTERVAL_S": "120", "CC_WANT_SKIP_SEEN_MAX": "7",
                             "CC_WANT_SKIP_DETAIL_PER_CALL_MAX": "3"}) == ["120", "7", "3"]
    # junk falls back to the default; below-minimum clamps to the minimum -- never breaks import
    assert _import_with_env({"CC_WANT_SKIP_SEEN_MAX": "not-a-number"})[1] == "4096"
    assert _import_with_env({"CC_WANT_SKIP_DETAIL_PER_CALL_MAX": "0"})[2] == "1"


def test_flood_claim_beyond_seen_max_degrades_to_the_per_call_cap(monkeypatch, caplog, clock):
    """le-014 C4, stated exactly: <= SEEN_MAX distinct skips -> quiet after draining;
    > SEEN_MAX -> evicted keys re-qualify, so up to DETAIL_PER_CALL_MAX lines per pulse, never more."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    monkeypatch.setattr(org, "WANT_SKIP_SEEN_MAX", 10)
    monkeypatch.setattr(org, "WANT_SKIP_DETAIL_PER_CALL_MAX", 5)
    g, vdb = _graph_with(*["stray [/WANT] %d" % i for i in range(30)])
    per_pulse = []
    for _ in range(12):
        before = len([l for l in _info(caplog) if "node=" in l])
        clock.now += 60
        org.surface_wants(g, vdb)
        per_pulse.append(len([l for l in _info(caplog) if "node=" in l]) - before)
    assert max(per_pulse) <= 5
    assert per_pulse[-1] == 5                        # does NOT go quiet: the documented degraded regime
    org._reset_want_skip_log_state()
    monkeypatch.setattr(org, "WANT_SKIP_SEEN_MAX", 4096)
    g, vdb = _graph_with(*["stray [/WANT] %d" % i for i in range(12)])
    tail = []
    for _ in range(6):
        before = len([l for l in _info(caplog) if "node=" in l])
        clock.now += 60
        org.surface_wants(g, vdb)
        tail.append(len([l for l in _info(caplog) if "node=" in l]) - before)
    assert tail[-1] == 0                             # within SEEN_MAX it drains and goes quiet


def test_log_docstring_no_longer_claims_no_marker_text():
    doc = org._log_want_skips.__doc__
    assert "NEVER logs marker" not in doc
    assert "literal marker token" in doc


# ---------------------------------------------------------------------------
# #815 (Exec P414): in_json_string / in_url / in_link_target
# A want tag inside JSON, a URL or a link target is text being CARRIED or TALKED ABOUT.
# The reason applies to the context the OPENER sits in -- never to what the want text contains.
# ---------------------------------------------------------------------------

_LONG_WANT = ("I genuinely want the parser to keep every sentence of a long forward intention intact. " * 9).strip()

# checker-016's exact examples -> (content, opener reason, closer reason)
_815_EXAMPLES = {
    "json object value": (
        '{"cmd":"[WANT] not a want [/WANT]"}', "in_json_string", "in_json_string"),
    "json-escaped quotes": (
        '\\"[WANT]\\" then later \\"[/WANT]\\"', "in_json_string", "closer_without_opener"),
    "tilde fence inside a JSON string (literal backslash-n)": (
        '{"code": "~~~\\n[WANT] documented [/WANT]\\n~~~"}', "in_json_string", "in_json_string"),
    "backtick fence inside a JSON string (literal backslash-n)": (
        '{"code": "```\\n[WANT] documented [/WANT]\\n```"}', "in_json_string", "in_json_string"),
    "URL path": (
        "https://example.com/path/[WANT]secret-want[/WANT]/docs", "in_url", "closer_without_opener"),
    "markdown link target": (
        "[the docs](https://example.com/[WANT]linked[/WANT])", "in_link_target", "closer_without_opener"),
}


def test_815_reasons_are_registered():
    for reason in ("in_json_string", "in_url", "in_link_target"):
        assert reason in org.WANT_SKIP_REASONS


@pytest.mark.parametrize("name", sorted(_815_EXAMPLES))
def test_815_example_is_rejected_with_its_reason(base_org, name):
    content, opener_reason, closer_reason = _815_EXAMPLES[name]
    parsed = org.parse_wants(content)
    assert parsed.wants == ()
    assert [(s.marker, s.reason) for s in parsed.skipped] == [("[WANT]", opener_reason), ("[/WANT]", closer_reason)]
    g, vdb = _graph_with(content)
    assert org.surface_wants(g, vdb) == [] and _want_nodes(g) == {}
    # a deliberate delta: base minted a (bogus) want from every one of these
    assert _texts(_run(base_org, [content])[0]) != []


@pytest.mark.parametrize("order", ["after", "before"])
@pytest.mark.parametrize("sep", ["\n\n", " "], ids=["new-paragraph", "same-line"])
@pytest.mark.parametrize("name", sorted(_815_EXAMPLES))
def test_815_genuine_long_want_in_plain_prose_beside_each_is_captured_whole(name, sep, order):
    content, opener_reason, closer_reason = _815_EXAMPLES[name]
    real = "[WANT]" + _LONG_WANT + "[/WANT]"
    text = content + sep + real if order == "after" else real + sep + content
    assert len(_LONG_WANT) > 600
    parsed = org.parse_wants(text)
    assert [w.text for w in parsed.wants] == [_LONG_WANT]          # whole, > 600 chars
    assert sorted((s.marker, s.reason) for s in parsed.skipped) == sorted(
        [("[WANT]", opener_reason), ("[/WANT]", closer_reason)])    # and the example is still skipped
    g, vdb = _graph_with(text)
    assert _texts(org.surface_wants(g, vdb)) == [_LONG_WANT]


def test_815_backtick_fence_rejection_does_not_depend_on_the_inline_code_coincidence():
    content = _815_EXAMPLES["backtick fence inside a JSON string (literal backslash-n)"][0]
    # the coincidence checker-016 noticed: the two ``` runs pair up as an inline code span...
    fences = org._want_fence_spans(content)
    spans = org._want_code_span_ranges(content, fences)
    opener = content.index("[WANT]")
    assert any(a <= opener < b for a, b in spans)
    # ...but the rejection is by JSON structure, which is checked FIRST
    assert {s.reason for s in org.parse_wants(content).skipped} == {"in_json_string"}
    # same content with the JSON wrapper broken (not a string literal) falls back to the old reason
    broken = content.replace('"code": ', "code = ")
    assert "in_json_string" not in {s.reason for s in org.parse_wants(broken).skipped}


# The BOUNDARY: context of the OPENER, not the contents of the want. (content, wants, skip reasons)
_815_BOUNDARY = {
    "want text that CONTAINS a URL": (
        "[WANT]please check https://example.com/path/ for x[/WANT]",
        ["please check https://example.com/path/ for x"], []),
    "want text that contains JSON": (
        '[WANT]add {"cmd": "run"} support to the parser[/WANT]',
        ['add {"cmd": "run"} support to the parser'], []),
    "want text that contains a markdown link": (
        "[WANT]read [the docs](https://example.com/x) first[/WANT]",
        ["read [the docs](https://example.com/x) first"], []),
    "a quoted word INSIDE the want, real tags in prose": (
        'He said "ship it" so [WANT]ship "it" today[/WANT]', ['ship "it" today'], []),
    "URL followed by a want in the same paragraph": (
        "see https://example.com/docs [WANT]follow up[/WANT]", ["follow up"], []),
    "URL then sentence dot glued to the opener": (
        "see https://example.com/docs.[WANT]follow up[/WANT]", ["follow up"], []),
    "parenthesised URL glued to the opener": (
        "(https://example.com/docs)[WANT]follow up[/WANT]", ["follow up"], []),
    "link glued to the opener": (
        "[t](https://x.org/a)[WANT]follow up[/WANT]", ["follow up"], []),
    "escaped quotes in prose": (
        'He typed \\"go\\" and then [WANT]follow up[/WANT]', ["follow up"], []),
    "prose quote followed by a comma": (
        'He said "go [WANT]x[/WANT] now", then left', ["x"], []),
    "a want after a quoted list": (
        'Items: "a", "b", [WANT]follow up[/WANT]', ["follow up"], []),
    "want that ENDS in a URL": (
        "[WANT]read https://example.com/docs[/WANT]", ["read https://example.com/docs"], []),
    "want that ends in a link": (
        "[WANT]read [d](https://x.org)[/WANT]", ["read [d](https://x.org)"], []),
    "a closer inside a JSON-looking literal with a REAL opener pending is the want's closer (turn 3)": (
        '[WANT]use {"k": "[/WANT]"} form[/WANT]', ['use {"k": "'], ["closer_without_opener"]),
}


@pytest.mark.parametrize("name", sorted(_815_BOUNDARY))
def test_815_boundary_reason_applies_to_the_openers_context_only(name):
    content, expected_wants, expected_reasons = _815_BOUNDARY[name]
    parsed = org.parse_wants(content)
    assert [w.text for w in parsed.wants] == expected_wants
    assert [s.reason for s in parsed.skipped] == expected_reasons


# JSON-looking contexts reject (pretty-printed, array element, inside a sentence).
_815_JSON_CONTEXTS = {
    "pretty-printed object": '{\n  "cmd": "[WANT] x [/WANT]"\n}',
    "array element": '["a", "[WANT] x [/WANT]"]',
    "legitimate want typed inside a JSON-looking sentence (KNOWN false negative)":
        'Use {"note": "[WANT] revisit X [/WANT]"} for it',
}


@pytest.mark.parametrize("name", sorted(_815_JSON_CONTEXTS))
def test_815_json_contexts_are_rejected_and_reported(name):
    parsed = org.parse_wants(_815_JSON_CONTEXTS[name])
    assert parsed.wants == ()
    assert {s.reason for s in parsed.skipped} == {"in_json_string"}


# What the delta deliberately does NOT recognise (pinned, so a future change is a visible decision).
_815_NOT_RECOGNISED = {
    "single-quoted JSON-ish": ("{'cmd': '[WANT] x [/WANT]'}", ["x"]),
    "markdown link TEXT containing a pair": ("[[WANT]text[/WANT]](https://x.org)", ["text"]),
    "a 'JSON string' with a raw newline is not JSON": ('{"cmd": "[WANT] x\n [/WANT]"}', ["x"]),
    "reference-style link definition without a scheme": ("[id]: ./dir/[WANT]x[/WANT]", ["x"]),
    "bare prose mention that pairs cleanly": ("use [WANT] to mark one, and [/WANT] closes it",
                                               ["to mark one, and"]),
}


@pytest.mark.parametrize("name", sorted(_815_NOT_RECOGNISED))
def test_815_not_recognised_shapes_stay_real_markers(name):
    content, expected = _815_NOT_RECOGNISED[name]
    assert [w.text for w in org.parse_wants(content).wants] == expected


def test_815_new_reasons_flow_through_the_flood_safe_log_unchanged(caplog, clock):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    g, vdb = _graph_with(_815_EXAMPLES["json object value"][0],
                         _815_EXAMPLES["URL path"][0],
                         _815_EXAMPLES["markdown link target"][0])
    org.surface_wants(g, vdb)
    lines = _info(caplog)
    summary = [l for l in lines if "marker(s) in" in l]
    assert len(summary) == 1
    for frag in ("in_json_string=2", "in_url=1", "in_link_target=1", "closer_without_opener=2"):
        assert frag in summary[0], summary[0]
    assert all("secret-want" not in r.getMessage() and "not a want" not in r.getMessage()
               for r in caplog.records)                             # no body / surrounding text
    before = len(lines)
    for _ in range(10):
        clock.now += 60
        org.surface_wants(g, vdb)
    assert len(_info(caplog)) == before                             # flood-safe form unchanged: quiet


def test_815_finders_are_linear_on_adversarial_input():
    import time as _time
    for content in ('"' * 200_000 + "[WANT] x [/WANT]",
                    '{"a":"' * 20_000 + "[WANT] x [/WANT]",
                    "a" * 100_000 + "[WANT] x [/WANT]",
                    ("https://x.org/" * 5_000) + "[WANT] x [/WANT]"):
        t0 = _time.perf_counter()
        org.parse_wants(content)
        assert _time.perf_counter() - t0 < 5.0


# ---------------------------------------------------------------------------
# TURN 3 (le-016 MEDIUM #2/#3, checker-018 notes 1-2): the lexical-guess false negatives, fixed
# ONCE. Every "must mint again" case is ASSERTED AGAINST BASE e4ebf982 (git show; missing = FAIL).
# ---------------------------------------------------------------------------

_T3_PAYLOAD = "follow up on the recall path"


def _assert_mints_like_base(base_org, content, expected):
    assert _texts(_run(base_org, [content])[0]) == expected, "the expectation must be what BASE mints"
    assert _run(org, [content]) == _run(base_org, [content])
    assert _texts(_run(org, [content])[0]) == expected


# --- (1) the CLOSER rule: a closer is judged relative to its OPENER, never by its own text ---
_T3_CLOSER_GOLDEN = {
    "le-016: want text ends inside a quote fragment":
        ('[WANT]rename "a", "b[/WANT]", next', ['rename "a", "b']),
    "fuzz-derived: key/value fragment": ('[WANT]set "x": "y[/WANT]", done', ['set "x": "y']),
    "fuzz-derived: list fragment": ('[WANT]list 1, "b[/WANT]", c', ['list 1, "b']),
    "fuzz-derived: brace fragment": ('[WANT]use {"a": "b[/WANT]"} now', ['use {"a": "b']),
    "fuzz-derived: array fragment": ('[WANT]pick ["a", "b[/WANT]"]', ['pick ["a", "b']),
    "fuzz-derived: two quote fragments": ('[WANT]use "a","b[/WANT]","c"', ['use "a","b']),
    "closer in a FENCE, real opener outside": ("[WANT] a\n```\n[/WANT]\n```\n b [/WANT]", ["a\n```"]),
    "closer in a CODE SPAN, real opener outside": ("[WANT] replace `[/WANT]` tokens [/WANT]", ["replace `"]),
    "closer in a stray-backtick span (le-014 C3, second shape)": ("[WANT]fix `a[/WANT] and `b` here", ["fix `a"]),
    "closer wrapped in a quote pair": ('x "[WANT]I want "x"[/WANT]" y', ['I want "x"']),
    "closer after an odd backslash": ("[WANT]path C:\\[/WANT]", ["path C:\\"]),
    "closer glued to a URL path": ("[WANT]read https://x.org/a/[/WANT]", ["read https://x.org/a/"]),
    "closer glued to a link destination": ("[WANT]read [d](https://x.org/a[/WANT]", ["read [d](https://x.org/a"]),
    "closer right after inline code": ("[WANT]check `foo()`[/WANT]", ["check `foo()`"]),
}


@pytest.mark.parametrize("name", sorted(_T3_CLOSER_GOLDEN))
def test_t3_closer_judged_only_relative_to_its_opener_equals_base(base_org, name):
    content, expected = _T3_CLOSER_GOLDEN[name]
    _assert_mints_like_base(base_org, content, expected)


def test_t3_a_closer_is_a_mention_only_with_a_mention_opener_in_its_region_or_no_opener():
    """The rule, both halves, one assertion each."""
    # half 1: a mention PAIR inside one region -> both masked (region reason), the real want keeps them
    for content, reason in (('{"cmd":"[WANT] x [/WANT]"}', "in_json_string"),
                            ("```\n[WANT] x [/WANT]\n```", "in_fence"),
                            ("`[WANT] x [/WANT]`", "in_code_span")):
        parsed = org.parse_wants(content)
        assert parsed.wants == () and [s.reason for s in parsed.skipped] == [reason, reason], content
    # half 2a: no live opener pending -> a closer inside a region is a mention, with the region's reason
    for content, reason in (('{"cmd":"x [/WANT]"}', "in_json_string"), ("```\nx [/WANT]\n```", "in_fence"),
                            ("`x [/WANT]`", "in_code_span")):
        parsed = org.parse_wants(content)
        assert parsed.wants == () and [s.reason for s in parsed.skipped] == [reason], content
    # half 2b: a REAL opener pending -> the closer pairs with it, whatever surrounds the closer
    for content, text in (('[WANT]a "b[/WANT]", c', 'a "b'), ("[WANT]a `b[/WANT]` c", "a `b"),
                          ("[WANT]a\n```\nb[/WANT]\n```\n", "a\n```\nb")):
        assert [w.text for w in org.parse_wants(content).wants] == [text], content


def test_t3_no_closer_is_judged_by_a_guess_on_its_own_surroundings():
    """AUDIT: each adjacency guess is opener-only. For every guess, a CLOSER in that exact
    context (real opener in plain prose) pairs; the same context on an OPENER is still a mention."""
    closer_contexts = {
        "quoted": '[WANT]say "x"[/WANT]" y',                       # closer hugged by a quote pair
        "escaped": "[WANT]path C:\\[/WANT] z",                     # odd backslash before the closer
        "code_adjacent": "[WANT]check `x`[/WANT] z",               # backtick right before the closer
        "in_url": "[WANT]read https://x.org/a/[/WANT] z",          # closer glued to a URL path
        "in_link_target": "[WANT]read [d](https://x.org/a[/WANT] z",
        "json-escaped hug": '[WANT]typed \\"go\\"[/WANT]\\" z',
    }
    for guess, content in closer_contexts.items():
        parsed = org.parse_wants(content)
        assert len(parsed.wants) == 1 and parsed.skipped == (), (guess, parsed)
    opener_contexts = {
        "quoted": '"[WANT]" x', "escaped": "\\[WANT] x", "code_adjacent": "a`[WANT] x",
        "in_url": "https://x.org/a/[WANT] x", "in_link_target": "[t](./d/[WANT] x",
        "in_json_string": '\\"[WANT]\\" x',
    }
    for reason, content in opener_contexts.items():
        parsed = org.parse_wants(content)
        assert parsed.skipped and parsed.skipped[0] == org.SkippedMarker("[WANT]", parsed.skipped[0].start, reason), (reason, parsed)


# --- (2) URL glue: only genuine URL-internal glue drops an opener ---
_T3_URL_MINTS = {
    "colon after the URL": "See https://x.org/a:[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "em dash": "See https://x.org/a\u2014[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "en dash": "See https://x.org/a\u2013[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "bold URL": "See **https://x.org/a**[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "italic URL": "_https://x.org/a_[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "tilde": "See https://x.org/a~[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "literal backslash-n (JSON-escaped text)": "See https://x.org/a\\n[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "sentence dot": "See https://x.org/a.[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "closing paren": "See (https://x.org/a)[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "comma": "See https://x.org/a,[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "hyphen": "See https://x.org/a-[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "? with the want AFTER the URL (closer ends the node)": "See https://x.org/a?[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "# with the want AFTER the URL": "See https://x.org/a#[WANT]%s[/WANT]" % _T3_PAYLOAD,
    "? with the want AFTER the URL (closer then space)": "See https://x.org/a?[WANT]%s[/WANT] ok" % _T3_PAYLOAD,
    "? with the want AFTER the URL (closer then sentence dot)": "See https://x.org/a?[WANT]%s[/WANT]." % _T3_PAYLOAD,
}


@pytest.mark.parametrize("name", sorted(_T3_URL_MINTS))
def test_t3_url_glue_that_is_not_url_internal_mints_like_base(base_org, name):
    _assert_mints_like_base(base_org, _T3_URL_MINTS[name], [_T3_PAYLOAD])


# the REJECTED set (deliberate deltas vs base): (content, opener reason). `/` always; `? # = &` only
# when the marker is INSIDE the URL token -- URL characters continue right after the closing tag.
_T3_URL_REJECTS = {
    "slash: checker-016's example (kept)": ("https://example.com/path/[WANT]secret-want[/WANT]/docs", "in_url"),
    "slash at the end of the node": ("https://example.com/path/[WANT]secret-want[/WANT]", "in_url"),
    "? INSIDE the token (query continues)": ("https://x.org/a?[WANT]secret[/WANT]&p=1", "in_url"),
    "# INSIDE the token (fragment continues)": ("https://x.org/a#[WANT]secret[/WANT]section", "in_url"),
    "= INSIDE the token (query value)": ("https://x.org/s?q=[WANT]secret[/WANT]&p=1", "in_url"),
    "& INSIDE the token": ("https://x.org/s?p=1&[WANT]secret[/WANT]=2", "in_url"),
}


@pytest.mark.parametrize("name", sorted(_T3_URL_REJECTS))
def test_t3_url_internal_glue_still_rejects_the_opener(base_org, name):
    content, reason = _T3_URL_REJECTS[name]
    parsed = org.parse_wants(content)
    assert parsed.wants == ()
    assert parsed.skipped[0].reason == reason and parsed.skipped[0].marker == "[WANT]"
    assert _texts(_run(base_org, [content])[0]) != []          # a deliberate delta: base minted it


def test_t3_url_rule_is_an_allowlist_not_a_blocklist():
    assert org._WANT_URL_PATH_GLUE == "/"
    assert set(org._WANT_URL_QUERY_GLUE) == set("?#=&")
    assert not hasattr(org, "_WANT_URL_TERMINATORS")


# --- (3) token boundary for true / false / null ---
_T3_TOKEN_MINTS = {
    "untrue": 'The claim is untrue, "[WANT]revisit this[/WANT]"',
    "nonnull": 'The pointer is nonnull, "[WANT]check it[/WANT]"',
    "intrue": 'It is intrue, "[WANT]check it[/WANT]"',
    "notfalse": 'notfalse, "[WANT]check it[/WANT]"',
    "underscore word": 'my_true, "[WANT]check it[/WANT]"',
    "digit-glued word": '2null, "[WANT]check it[/WANT]"',
}


@pytest.mark.parametrize("name", sorted(_T3_TOKEN_MINTS))
def test_t3_words_that_merely_end_in_true_false_null_do_not_open_a_json_literal(base_org, name):
    content = _T3_TOKEN_MINTS[name]
    expected = [content.split("[WANT]")[1].split("[/WANT]")[0]]
    _assert_mints_like_base(base_org, content, expected)


def test_t3_real_json_tokens_inside_a_container_still_count(base_org):
    """turn 4 (le-019 F2): this was the turn-3 'named residual' (`ok: true, "..."` dropped). The
    trigger was NOT limited to JSON-looking sentences -- ANY word true/false/null or digit run +
    comma + quote dropped a well-formed want -- so it is FIXED, not named: a true/false/null/number
    before the comma counts only inside a REAL JSON array/object. Real containers still reject."""
    for word in ("true", "false", "null", "1", "-1.5e3"):
        for content in ('[%s, "[WANT]revisit this[/WANT]"]' % word,
                        '{"a": %s, "[WANT]revisit this[/WANT]": 2}' % word):
            parsed = org.parse_wants(content)
            assert parsed.wants == () and {s.reason for s in parsed.skipped} == {"in_json_string"}, content
    # ...and the same words in ordinary prose / YAML-ish text mint like base
    for word in ("true", "false", "null"):
        _assert_mints_like_base(base_org, 'ok: %s, "[WANT]revisit this[/WANT]"' % word, ["revisit this"])


# --- (4) in_link_target needs a real markdown link: a `[` that opens it ---
_T3_LINK_NOT_A_LINK = {     # relative destinations: NOT a link target, no URL scheme -> mints like base
    "array index": "arr[0](./dir/[WANT]x[/WANT])",
    "call then index": "f(x)[0](./dir/[WANT]x[/WANT])",
    "snake_case index": "foo_bar[i](./dir/[WANT]x[/WANT])",
    "chained index": "a[b][c](./dir/[WANT]x[/WANT])",
    "no `[` in the paragraph": "unclosed](./dir/[WANT]x[/WANT])",
    "link text broken by a blank line": "[unclosed\n\ntext](./dir/[WANT]x[/WANT])",
}
_T3_LINK_IS_A_LINK = {
    "plain link": "[text](https://x.org/[WANT]x[/WANT])",
    "link with spaces in the text": "see [the docs](./dir/[WANT]x[/WANT])",
    "bold link": "**[t](./u/[WANT]x[/WANT])**",
    "image": "![alt text](./u/[WANT]x[/WANT])",
    "nested brackets in the text": "[a [b] c](./u/[WANT]x[/WANT])",
    "link after punctuation": "(see [t](./u/[WANT]x[/WANT]))",
}


@pytest.mark.parametrize("name", sorted(_T3_LINK_NOT_A_LINK))
def test_t3_a_bracket_paren_pair_that_is_not_a_link_is_not_a_link_target(base_org, name):
    _assert_mints_like_base(base_org, _T3_LINK_NOT_A_LINK[name], ["x"])


@pytest.mark.parametrize("name", sorted(_T3_LINK_IS_A_LINK))
def test_t3_a_real_markdown_link_destination_is_still_in_link_target(base_org, name):
    content = _T3_LINK_IS_A_LINK[name]
    parsed = org.parse_wants(content)
    assert parsed.wants == () and parsed.skipped[0].reason == "in_link_target"
    assert _texts(_run(base_org, [content])[0]) == ["x"]       # deliberate delta: base minted it


def test_t3_index_then_a_url_destination_is_not_a_link_target_but_is_still_a_url():
    """checker-018 note 2, stated exactly: `arr[0](https://x.org/[WANT]x[/WANT])` is NOT a link target
    (the `[` is an index); the opener is nevertheless glued to a URL PATH (`https://x.org/`), so the
    URL rule (not the link rule) still rejects it. The scheme-less sibling mints."""
    parsed = org.parse_wants("arr[0](https://x.org/[WANT]x[/WANT])")
    assert parsed.wants == () and parsed.skipped[0].reason == "in_url"
    assert [w.text for w in org.parse_wants("arr[0](./dir/[WANT]x[/WANT])").wants] == ["x"]


# --- named residuals that STILL behave as documented (pinned) ---
def test_t3_named_residual_unclosed_fence_glued_to_a_closer_swallows_later_wants(base_org):
    """A want whose text ends with a fence-closing line GLUED to the closer never closes that fence
    (CommonMark: a closing fence carries nothing after it), so the fence runs to the end of the node
    and a later want is masked in_fence. The first want now mints (turn-3 closer rule); base also
    minted the second. Documented failure mode 2, unchanged."""
    content = "[WANT]ends with a fence\n```\ncode\n```[/WANT] [WANT]second[/WANT]"
    assert _texts(_run(base_org, [content])[0]) == ["ends with a fence\n```\ncode\n```", "second"]
    parsed = org.parse_wants(content)
    assert [w.text for w in parsed.wants] == ["ends with a fence\n```\ncode\n```"]
    assert [(s.marker, s.reason) for s in parsed.skipped] == [("[WANT]", "in_fence"), ("[/WANT]", "in_fence")]


# --- (5) the changelog no longer over-claims ---
def test_t3_changelog_claim_is_qualified_to_the_tested_grammar():
    header = (_WORKTREE / "cc_ng_organism.py").read_text(encoding="utf-8").split("# -------------------")[0]
    assert "TESTED grammar" in header and "NOT for every string" in header


# --- cost: the new lookbacks stay bounded on adversarial input ---
def test_t3_link_and_url_scanners_stay_bounded_on_adversarial_input():
    import time as _time
    for content in (("[t](https://x.org/[WANT]x[/WANT]) " * 3_000),
                    ("[" * 50_000 + "](./d/[WANT]x[/WANT])"),
                    ("a[0](./d/[WANT]x[/WANT]) " * 3_000)):
        t0 = _time.perf_counter()
        org.parse_wants(content)
        assert _time.perf_counter() - t0 < 10.0


# ---------------------------------------------------------------------------
# TURN 4 (le-019 F1-F5): the NEW final function. Golden cases ASSERTED AGAINST BASE e4ebf982.
# ---------------------------------------------------------------------------

def _inner(content):
    return content.split("[WANT]", 1)[1].split("[/WANT]", 1)[0]


# --- F2: a bare prose word / number + comma + quote is NOT a JSON literal ---
_T4_F2_MINTS = {
    "true": 'It is true, "[WANT]revisit this[/WANT]", she said.',
    "false": 'That is false, "[WANT]revisit this[/WANT]", ok',
    "null": 'if null, "[WANT]revisit this[/WANT]", ok',
    "year (le-019)": 'In 2026, "[WANT]revisit the exit policy[/WANT]", I wrote.',
    "chapter number": 'Chapter 3, "[WANT]revisit this[/WANT]"',
    "number": 'we saw 12, "[WANT]revisit this[/WANT]", ok',
    "minus true": '-true, "[WANT]revisit this[/WANT]"',
    "dot null": '.null, "[WANT]revisit this[/WANT]"',
    "digit glued to comma quote (le-016)": '3,"[WANT]x[/WANT]",done',
    "quoted phrase list in prose (le-016)": 'She said "a", "b [WANT]x[/WANT]", and left.',
    "YAML-ish ok: true": 'ok: true, "[WANT]revisit this[/WANT]"',
    "the answer is true": 'the answer is true, "[WANT]revisit this[/WANT]", yes',
    "it was 3": 'it was 3, "[WANT]revisit this[/WANT]", yes',
    "number then brace closer": 'we saw 12, "[WANT]revisit this[/WANT]"}',
    "word then bracket closer": 'see two, "[WANT]revisit this[/WANT]"]',
}


@pytest.mark.parametrize("name", sorted(_T4_F2_MINTS))
def test_t4_f2_prose_word_or_number_comma_quote_mints_like_base(base_org, name):
    content = _T4_F2_MINTS[name]
    _assert_mints_like_base(base_org, content, [_inner(content)])


# Real JSON context (a structural opener `{` / `[` reached by walking back over complete values
# and `"key":` members) stays rejected: checker-016's examples, le-016's and checker-018's.
_T4_JSON_REJECTS = {
    "checker-016 object value": '{"cmd":"[WANT] not a want [/WANT]"}',
    "checker-016 escaped quotes": '\\"[WANT]\\" then later \\"[/WANT]\\"',
    "checker-016 tilde fence in JSON": '{"code": "~~~\\n[WANT] documented [/WANT]\\n~~~"}',
    "checker-016 backtick fence in JSON": '{"code": "```\\n[WANT] documented [/WANT]\\n```"}',
    "JSON string list": '["a", "[WANT] x [/WANT]"]',
    "object: true then a key": '{"a": true, "[WANT]x[/WANT]": 1}',
    "array: true first": '[true, "[WANT]x[/WANT]"]',
    "object: number then a key": '{"a": 1, "[WANT]x[/WANT]": 2}',
    "array of numbers": '[1, 2, "[WANT]x[/WANT]"]',
    "array: float": '[-1.5e3, "[WANT]x[/WANT]"]',
    "array: nested array first": '[["a"], "[WANT]x[/WANT]"]',
    "array: nested object first": '[{"k": 1}, "[WANT]x[/WANT]"]',
    "array: null first": '[null, "[WANT]x[/WANT]"]',
    "pretty-printed array": '[\n  "a",\n  "[WANT]x[/WANT]"\n]',
    "pretty-printed object": '{\n  "a": 1,\n  "b": "[WANT]x[/WANT]"\n}',
    "object: several members": '{"a": 1, "b": 2, "c": "[WANT]x[/WANT]"}',
    "object: array member then key": '{"a": [1, 2], "[WANT]x[/WANT]": 3}',
    "known false negative: a want inside {\"note\": ...}": 'Use {"note": "[WANT] revisit X [/WANT]"} for it',
}


@pytest.mark.parametrize("name", sorted(_T4_JSON_REJECTS))
def test_t4_f2_real_json_context_is_still_rejected(name):
    parsed = org.parse_wants(_T4_JSON_REJECTS[name])
    assert parsed.wants == (), parsed
    assert {s.reason for s in parsed.skipped} <= {"in_json_string", "closer_without_opener"}
    assert "in_json_string" in {s.reason for s in parsed.skipped}


_T4_WALK_CASES = [      # (text ending in the candidate opening quote, expected "JSON opener context")
    ('x true, "', False), ('[true, "', True), ('{"a": true, "', True), ('it was 3, "', False),
    ('[1, 2, "', True), ('["a", "b", "', True), ('"a", "', False), ('ok: true, "', False),
    ('{"a": 1, "b": 2, "', True), ('[[1], "', True), ('{"a": [1, 2], "', True),
    ('{"a": "x", "', True), ('abc12, "', False), ('[untrue, "', False), ('word, "', False),
    ('1, "', False), ('{"a": 1 "', False), ('["a" "', False),
]


@pytest.mark.parametrize("text,expected", _T4_WALK_CASES)
def test_t4_f2_structure_walk_unit(text, expected):
    assert org._want_json_opens_literal(text, len(text) - 1) is expected
    assert org._want_json_opens_literal(text, len(text) - 1, {}) is expected       # memoised form agrees


_T4_PROSE_PREFIXES = ['It is true, "', 'In 2026, "', 'Chapter 3, "', 'we saw 12, "', 'if null, "',
                      'the answer is false, "', 'She said "a", "', "x, "]
_T4_PROSE_BODIES = ["plain want", "revisit this", "ends `code()`", 'has "inner" word', "with (parens) and [brackets]",
                    "multi\nline body", "see https://example.com/docs now", '{"a": 1} inside']
_T4_PROSE_SUFFIXES = ['", she said.', '",', '"', '"}', '"]', '", next', ' tail', ".", '",\n']


def test_t4_parity_prose_comma_quote_shapes_with_base(base_org):
    """The F2 regression net: {bare prose word/number/quoted phrase + comma + quote} x {body} x
    {JSON-closer-looking suffix}: the parser equals base on every one (none sits in a real JSON
    container, and no body starts with a quote, so no `"[WANT]"` hug is built)."""
    total = 0
    diverged = []
    for prefix, body, suffix in itertools.product(_T4_PROSE_PREFIXES, _T4_PROSE_BODIES, _T4_PROSE_SUFFIXES):
        content = prefix + "[WANT]" + body + "[/WANT]" + suffix
        total += 1
        if _run(org, [content]) != _run(base_org, [content]):
            diverged.append(content)
    assert total == len(_T4_PROSE_PREFIXES) * len(_T4_PROSE_BODIES) * len(_T4_PROSE_SUFFIXES) >= 500
    assert diverged == [], "diverged on %d/%d, first: %r" % (len(diverged), total, diverged[:3])


# --- F1: cost -- the `? # = &` branch is linear in the node ---
def test_t4_f1_url_query_glue_scan_is_linear_not_quadratic():
    import time as _time
    timings = {}
    for n in (5_000, 20_000, 40_000):
        content = ("https://x.org/a?[WANT]w " * n) + "[/WANT]-x"     # URL chars continue after the closer
        t0 = _time.perf_counter()
        parsed = org.parse_wants(content)
        timings[n] = _time.perf_counter() - t0
        # behaviour unchanged vs turn 3: every opener is in_url (inside the token), the closer is stray
        assert len(parsed.skipped) == n + 1 and parsed.wants == ()
        assert [s.reason for s in parsed.skipped].count("in_url") == n
    assert timings[40_000] < 8.0, timings                    # was 16.3 s (le-019); ~1.2 s now
    assert timings[40_000] < timings[5_000] * 20, timings    # 8x the input, far from the 64x of quadratic


def test_t4_f1_next_closer_is_a_bisect_over_the_precomputed_closer_list():
    content = "a?[WANT]x[/WANT]-tail [WANT]y[/WANT] z"
    closers = [content.index("[/WANT]"), content.rindex("[/WANT]")]
    first_end = content.index("[WANT]") + len("[WANT]")
    assert org._want_url_continues_after_pair(content, first_end, closers) is True       # `-` continues a URL
    second_end = content.rindex("[WANT]") + len("[WANT]")
    assert org._want_url_continues_after_pair(content, second_end, closers) is False     # space follows
    assert org._want_url_continues_after_pair(content, len(content), closers) is False   # no closer left


def test_t4_json_and_link_scanners_stay_bounded_on_adversarial_input():
    import time as _time
    for content in ('"a", ' * 40_000 + "[WANT] x [/WANT]",
                    '["a", ' * 40_000 + '"[WANT]x[/WANT]"]',
                    "1, " * 100_000 + '"[WANT]x[/WANT]"',
                    "{}, " * 50_000 + '"[WANT]x[/WANT]"',
                    '"' * 200_000 + "[WANT] x [/WANT]"):
        t0 = _time.perf_counter()
        org.parse_wants(content)
        assert _time.perf_counter() - t0 < 8.0


# --- F3: a mention opener nested in a real want pairs with the next closer (nearest opener) ---
_T4_F3_WHOLE = {       # base minted NOTHING (inner holds a marker); the build keeps the whole want, pair and all
    "URL-carried mention pair": ('[WANT]see https://x.org/[WANT]z[/WANT] ok[/WANT]',
                                 "see https://x.org/[WANT]z[/WANT] ok"),
    "link-carried mention pair": ('[WANT]see [t](https://x.org/[WANT]z[/WANT]) ok[/WANT]',
                                  "see [t](https://x.org/[WANT]z[/WANT]) ok"),
    "quoted mention pair": ('[WANT]use "[WANT]" to start and "[/WANT]" to end[/WANT]',
                            'use "[WANT]" to start and "[/WANT]" to end'),
    "escaped mention pair": ('[WANT]use \\[WANT] to start and \\[/WANT] to end[/WANT]',
                             'use \\[WANT] to start and \\[/WANT] to end'),
    "region mention pair (contrast: already whole since turn 3)": ('[WANT]see `[WANT]z[/WANT]` ok[/WANT]',
                                                                  "see `[WANT]z[/WANT]` ok"),
}


@pytest.mark.parametrize("name", sorted(_T4_F3_WHOLE))
def test_t4_f3_a_mention_pair_inside_a_real_want_stays_inside_it(base_org, name):
    """NOT a truncated marker-bearing mint (turn 3 minted `see https://x.org/[WANT]z` here): the want is
    kept WHOLE, because the nested mention opener pairs with the NEXT closer (nearest-opener rule) and the
    real opener with the one after. Base minted nothing (its inner text holds a marker): a deliberate P406 delta."""
    content, whole = _T4_F3_WHOLE[name]
    assert _texts(_run(base_org, [content])[0]) == []
    parsed = org.parse_wants(content)
    assert [w.text for w in parsed.wants] == [whole]
    assert len(parsed.skipped) % 2 == 0 and {s.marker for s in parsed.skipped} == {"[WANT]", "[/WANT]"}


def test_t4_f3_two_mention_pairs_in_one_real_want_stay_inside_it(base_org):
    """Base minted a garbage fragment here (its regex paired the real opener with the FIRST closer, then
    `d\\` from the second pair); the build keeps the whole want."""
    content = '[WANT]a \\[WANT]b\\[/WANT] c \\[WANT]d\\[/WANT] e[/WANT]'
    assert _texts(_run(base_org, [content])[0]) == ["d\\"]
    assert [w.text for w in org.parse_wants(content).wants] == ["a \\[WANT]b\\[/WANT] c \\[WANT]d\\[/WANT] e"]


_T4_F3_LIKE_BASE = {   # an UNPAIRED mention opener in a real want: its closer pairs with it -> nothing minted, as base
    "unpaired escaped mention": ('[WANT]use \\[WANT] to start[/WANT]', []),
    "unpaired quoted mention": ('[WANT]stop writing "[WANT]" by hand[/WANT]', []),
    "unpaired mention, then a second want": ('[WANT]use \\[WANT] ok[/WANT] later [WANT]second[/WANT]', ["second"]),
}


@pytest.mark.parametrize("name", sorted(_T4_F3_LIKE_BASE))
def test_t4_f3_an_unpaired_mention_yields_nothing_exactly_like_base(base_org, name):
    content, expected = _T4_F3_LIKE_BASE[name]
    _assert_mints_like_base(base_org, content, expected)
    assert "opener_unclosed" in {s.reason for s in org.parse_wants(content).skipped}    # visible, never silent


def test_t4_f3_mention_openers_outside_a_pending_want_are_unchanged():
    parsed = org.parse_wants('the "[WANT]" tag and "[/WANT]" end')
    assert parsed.wants == () and [s.reason for s in parsed.skipped] == ["quoted", "closer_without_opener"]
    assert [w.text for w in org.parse_wants('the "[WANT]" tag then [WANT]real[/WANT]').wants] == ["real"]


# --- F4: the complete named-residual list, pinned at CURRENT behaviour ---
_T4_F4A = {            # a previous want's OWN body starts a region that masks a LATER want's opener
    "unpaired backtick in the first want's body":
        ("[WANT]first has a ` tick[/WANT] then [WANT]second `code` here[/WANT]", ["first has a ` tick"]),
    "fence-looking line in the first want's body":
        ("[WANT]first\n~~~\nx[/WANT] then [WANT]second[/WANT]", ["first\n~~~\nx"]),
    "JSON-literal opener in the first want's body":
        ('[WANT]first "k": "v[/WANT] then [WANT]second[/WANT] x"}', ['first "k": "v']),
}


@pytest.mark.parametrize("name", sorted(_T4_F4A))
def test_t4_f4a_named_residual_a_previous_bodys_region_masks_a_later_opener(base_org, name):
    """KNOWN RESIDUAL: the FIRST want never diverges from base; a LATER want whose opener falls inside a
    region started in an earlier want's body is dropped (logged, in_code_span / in_fence / in_json_string)."""
    content, new_expected = _T4_F4A[name]
    base_texts = _texts(_run(base_org, [content])[0])
    assert len(base_texts) == 2 and base_texts[0] == new_expected[0]     # base minted both; the first is identical
    assert _texts(_run(org, [content])[0]) == new_expected
    assert {s.reason for s in org.parse_wants(content).skipped} & {"in_code_span", "in_fence", "in_json_string"}


def test_t4_f4bc_named_residuals_link_glued_to_a_word_and_a_url_char_after_a_link():
    # (b) a real link glued to a preceding word fails the `[`-neighbour rule; scheme-less, it mints
    assert [w.text for w in org.parse_wants("word[docs](./d/[WANT]x[/WANT])").wants] == ["x"]
    # (c) the glued run still holds the closed link's `scheme://`, and `/` is URL-internal glue
    parsed = org.parse_wants("See [t](https://x.org/a)/[WANT]x[/WANT]")
    assert parsed.wants == () and parsed.skipped[0].reason == "in_url"


def test_t4_f4_named_residual_a_json_container_with_a_very_long_element_is_not_recognised():
    """Fail-open toward minting: an element longer than the 512-character string window is not walked."""
    content = '["%s", "[WANT]x[/WANT]"]' % ("a" * 600)
    assert [w.text for w in org.parse_wants(content).wants] == ["x"]


# --- F5: doc ---
def test_t4_f5_turn2_url_terminator_rule_is_marked_superseded_and_turn4_entry_exists():
    header = (_WORKTREE / "cc_ng_organism.py").read_text(encoding="utf-8").split("# -------------------")[0]
    assert "SUPERSEDED BY TURN 3" in header
    assert "#810 turn 4" in header and "NEW FINAL FUNCTION" in header
