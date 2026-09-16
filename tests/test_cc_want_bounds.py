"""Bounds on CC want extraction and rendering (2026-09-16).

Regression cover for the "## What I Want" context bomb: 182 want-nodes, 118 of
them over 600 chars, largest 136,449, total 2.27 MB injected on every prompt.
Cause: prose that merely MENTIONS `[WANT]` let the unbounded non-greedy span run
to the next `[/WANT]` far away, and the swallowed text became a single want.
"""
import threading

import pytest

import cc_ng_organism as org


class _FakeNode:
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})
        self.creation_time = 0.0


class _FakeGraph:
    def __init__(self):
        self.nodes = {}
        self._step_lock = threading.RLock()   # the canonical lock surface_wants takes

    def create_node(self, node_id, metadata=None):
        n = _FakeNode(metadata)
        self.nodes[node_id] = n
        return n

    def create_synapse(self, a, b, weight=0.0):
        return None


class _FakeVDB:
    def __init__(self, content):
        self.content = content


def _graph_with(content_text):
    g = _FakeGraph()
    src = "cc:conv::src1"
    g.nodes[src] = _FakeNode({"creation_mode": "conversational"})
    return g, _FakeVDB({src: content_text})


def _texts(wants):
    return [w["text"] for w in wants]


def test_real_want_is_still_extracted():
    g, vdb = _graph_with("thinking out loud [WANT] follow up on the numpy/scipy "
                         "conflict later [/WANT] anyway")
    assert _texts(org.surface_wants(g, vdb)) == [
        "follow up on the numpy/scipy conflict later"]


def test_code_span_mention_creates_no_want():
    """The exact production shape: prose documenting the marker syntax."""
    content = (
        "The reaction loop dispatches on markers:\n"
        "- `[WANT]` -> write to the wants register\n"
        + "filler discussion of the architecture. " * 40 +
        "\nand the closing `[/WANT]` ends the span.\n"
    )
    g, vdb = _graph_with(content)
    assert org.surface_wants(g, vdb) == []


def test_span_longer_than_cap_creates_no_want():
    content = "[WANT]" + ("x" * (org.WANT_MAX_CHARS + 50)) + "[/WANT]"
    g, vdb = _graph_with(content)
    assert org.surface_wants(g, vdb) == []


def test_span_at_cap_is_accepted():
    body = "y" * org.WANT_MAX_CHARS
    g, vdb = _graph_with("[WANT]" + body + "[/WANT]")
    assert _texts(org.surface_wants(g, vdb)) == [body]


def test_nested_marker_is_rejected():
    g, vdb = _graph_with("[WANT] outer [WANT] inner [/WANT]")
    assert org.surface_wants(g, vdb) == []


def test_two_real_wants_both_extracted():
    g, vdb = _graph_with("[WANT] first thing [/WANT] noise [WANT] second thing [/WANT]")
    assert _texts(org.surface_wants(g, vdb)) == ["first thing", "second thing"]


def test_render_wants_caps_count_and_length():
    g = _FakeGraph()
    for i in range(org.WANT_RENDER_LIMIT + 15):
        n = _FakeNode({"kind": "want", "want_text": "w%03d " % i + "z" * 5000,
                       "want_state": "open", "provenance": "cc_authored"})
        n.creation_time = float(i)
        g.nodes["cc:want::%03d" % i] = n

    block = org.render_wants(g)
    lines = [l for l in block.split("\n") if l.startswith("- ")]
    assert len(lines) == org.WANT_RENDER_LIMIT + 1          # +1 elision line
    assert lines[-1].endswith("older open wants")
    assert max(len(l) for l in lines) <= org.WANT_MAX_CHARS + 2
    assert len(block) < 40_000                              # was 2.27 MB


def test_render_wants_under_limit_has_no_elision():
    g = _FakeGraph()
    n = _FakeNode({"kind": "want", "want_text": "a small want",
                   "want_state": "open", "provenance": "cc_authored"})
    g.nodes["cc:want::a"] = n
    assert org.render_wants(g) == "## What I Want\n- a small want"
