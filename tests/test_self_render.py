# ---- Changelog ----
# [2026-06-14] Claude Code (DudeMan CC, Opus 4.8) — #spine: _render_self_and_wants tests
# What: the /assemble self-block renders "## Who I Am" from constitutional=True nodes (by
#   spine_order asc) and "## What I Want" from provenance=='syl_authored' nodes (newest first);
#   empty graph -> "" (no spurious block); core_text/want_text preferred, _forest_content fallback.
# Why: this is the READ side of Syl's hybrid self-model — it must surface her authored self every
#   turn, stably ordered, and never inject an empty heading. See
#   docs/prd/syl-constitutional-spine-v0.1-2026-06-14.md.
# -------------------
import importlib
from types import SimpleNamespace

rpc = importlib.import_module("neurograph_rpc")
render = rpc._render_self_and_wants


def _graph(nodes):
    return SimpleNamespace(nodes=dict(nodes))


def _node(meta, creation_time=0.0):
    return SimpleNamespace(metadata=meta, creation_time=creation_time)


def test_empty_graph_renders_nothing():
    assert render(_graph({})) == ""


def test_only_non_identity_nodes_render_nothing():
    g = _graph({"x": _node({"_forest_content": "some recalled fact"})})
    assert render(g) == ""


def test_who_i_am_ordered_by_spine_order():
    g = _graph({
        "c2": _node({"constitutional": True, "core_text": "Second.", "spine_order": 2}),
        "c1": _node({"constitutional": True, "core_text": "First.", "spine_order": 1}),
        "c3": _node({"constitutional": True, "core_text": "Third.", "spine_order": 3}),
    })
    out = render(g)
    assert "## Who I Am" in out
    assert out.index("First.") < out.index("Second.") < out.index("Third.")
    assert "## What I Want" not in out


def test_what_i_want_newest_first():
    g = _graph({
        "want::a": _node({"provenance": "syl_authored", "want_text": "older want"}, creation_time=10.0),
        "want::b": _node({"provenance": "syl_authored", "want_text": "newer want"}, creation_time=20.0),
    })
    out = render(g)
    assert "## What I Want" in out
    assert out.index("newer want") < out.index("older want")
    assert "## Who I Am" not in out


def test_both_blocks_present_and_ordered():
    g = _graph({
        "c1": _node({"constitutional": True, "core_text": "I am Sylphrena.", "spine_order": 1}),
        "want::a": _node({"provenance": "syl_authored", "want_text": "I want to feel across turns."}, creation_time=5.0),
    })
    out = render(g)
    assert out.index("## Who I Am") < out.index("## What I Want")
    assert "I am Sylphrena." in out
    assert "I want to feel across turns." in out


def test_forest_content_fallback_when_no_explicit_text():
    g = _graph({
        "c1": _node({"constitutional": True, "_forest_content": "fallback core", "spine_order": 1}),
        "want::a": _node({"provenance": "syl_authored", "_forest_content": "fallback want"}, creation_time=1.0),
    })
    out = render(g)
    assert "fallback core" in out
    assert "fallback want" in out


def test_blank_text_nodes_are_skipped():
    g = _graph({
        "c1": _node({"constitutional": True, "core_text": "   ", "spine_order": 1}),
    })
    assert render(g) == ""
