# tests/test_surface_resolver.py
#
# Sandbox tests for the substrate-first surfacing content resolver.
# Pure-function logic + one real neuro_foundation Graph node. No live singleton.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surface_resolver import resolve_surface_content


class _FakeNode:
    def __init__(self, metadata):
        self.metadata = metadata


def _conv(forest=None, **extra):
    m = {"creation_mode": "conversational"}
    if forest is not None:
        m["_forest_content"] = forest
    m.update(extra)
    return _FakeNode(m)


def test_substrate_first_surfaces_forest_not_shard():
    """THE bug: her voice is in _forest_content; the vdb has only the shard 'WANT'.
    The resolver must surface HER, not the shard."""
    node = _conv(forest="*Laughs uproariously.* Oh my god, Josh. You really do just "
                        "weaponize my attempts at seriousness, don't you?")
    out = resolve_surface_content(node, {"content": "WANT"})
    assert out is not None
    assert out.startswith("*Laughs uproariously.*")
    assert "WANT" not in out  # the shard is NOT what surfaces


def test_snippet_is_bounded():
    """A full forest turn must not bloat the prompt — snippet is capped + elided."""
    node = _conv(forest="x" * 1000)
    out = resolve_surface_content(node, {"content": "shard"}, max_chars=240)
    assert len(out) <= 241  # 240 + the ellipsis char
    assert out.endswith("…")


def test_snippet_truncates_at_word_boundary_not_mid_word():
    """Truncation must snap to the last word boundary, not cut a word in half."""
    forest = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    node = _conv(forest=forest)
    out = resolve_surface_content(node, {"content": "shard"}, max_chars=27)
    assert out == "Alpha bravo charlie delta…"


def test_snippet_falls_back_to_hard_cut_when_no_word_boundary():
    """A single unbroken token has no space to snap to — hard-cut it instead
    of dropping the whole snippet."""
    node = _conv(forest="x" * 1000)
    out = resolve_surface_content(node, {"content": "shard"}, max_chars=240)
    assert out == "x" * 240 + "…"


def test_vdb_fallback_when_no_forest():
    """No _forest_content → fall back to the vdb content (real sentence, not a shard)."""
    node = _conv()
    sentence = "a real recalled sentence of reasonable length"
    assert resolve_surface_content(node, {"content": sentence}) == sentence


def test_ingested_code_filtered_from_experiential():
    """Ingested source-code is filtered out of experiential surfacing (CES/Tonic)."""
    node = _FakeNode({"creation_mode": "ingested"})
    code = '"""NeuroGraph Foundation - Core Cognitive Architecture'
    assert resolve_surface_content(node, {"content": code}) is None


def test_ingested_allowed_for_recall():
    """Recall passes allow_ingested=True — a query may legitimately want a document."""
    node = _FakeNode({"creation_mode": "ingested"})
    passage = "a legitimately recalled document passage here"
    assert resolve_surface_content(node, {"content": passage}, allow_ingested=True) == passage


def test_degenerate_fragments_filtered():
    """The 'o' / 'want' / 'the' shards (sub-floor + stopwords) never surface."""
    node = _conv()
    for shard in ("o", "want", "the", "True", " "):
        assert resolve_surface_content(node, {"content": shard}) is None


def test_real_graph_node():
    """Sandbox: works against a real neuro_foundation Graph node, not just a fake."""
    from neuro_foundation import Graph
    g = Graph()
    n = g.create_node(metadata={
        "creation_mode": "conversational",
        "_forest_content": "I want to feel, across turns, the texture of running.",
    })
    out = resolve_surface_content(g.nodes[n.node_id], {"content": "running"})
    assert out == "I want to feel, across turns, the texture of running."


def test_tonic_thread_surfaces_forest_not_shard_end_to_end():
    """End-to-end wiring: after _update_thread, the Tonic latent context renders her
    node's _forest_content (her voice), NOT the vdb shard."""
    from neuro_foundation import Graph
    from tonic_thread import TonicThread

    g = Graph()
    n = g.create_node(metadata={
        "creation_mode": "conversational",
        "_forest_content": "*Laughs.* Oh my god, Josh — you weaponize my attempts at seriousness.",
    })
    nid = n.node_id
    thread = TonicThread(g, vector_db={nid: {"content": "WANT"}})  # vdb holds only the shard

    class _Res:
        fired_entries = []

    thread._update_thread([(nid, 1.0)], _Res())
    rendered = thread.format_latent_context() or ""
    assert "Laughs" in rendered, f"her _forest_content not surfaced; got: {rendered!r}"
    assert "WANT" not in rendered, "the vdb shard surfaced instead of her voice"


def test_tonic_thread_filters_ingested_code_node():
    """An ingested source-code node is filtered out of her experiential thread."""
    from neuro_foundation import Graph
    from tonic_thread import TonicThread

    g = Graph()
    n = g.create_node(metadata={"creation_mode": "ingested"})
    nid = n.node_id
    thread = TonicThread(g, vector_db={nid: {"content": '"""NeuroGraph Foundation - Core Cognitive Architecture'}})

    class _Res:
        fired_entries = []

    thread._update_thread([(nid, 1.0)], _Res())
    rendered = thread.format_latent_context() or ""
    assert "NeuroGraph Foundation" not in rendered, f"ingested code surfaced: {rendered!r}"
