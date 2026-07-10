# ---- Changelog ----
# [2026-07-06] Claude Code (Sonnet 5) — Pattern-completion recall tests
# What: Tests for cc_pattern_completion_recall(), _format_cc_recall_block(), gate_pattern_completion().
# Why:  docs/prd/2026-07-06-cc-surfacing-pattern-completion-tier-drop.md.
# How:  Real NeuroGraphMemory fixture (same pattern as test_cc_dual_pass.py's cc_ng fixture) for the
#       recall/formatting tests; plain-dict tests for the pure gate function.
# [2026-07-07] Claude Code (Fable 5) — #358 rebuild: adjust two tests to the new
#   substrate-native recall contract (Task 4)
# What: test_pattern_completion_recall_surfaces_aged_out_content now lowers the
#   deposited node's threshold for deterministic firing (recall requires the
#   node to actually fire post-rebuild, not just be cosine-similar).
#   test_pattern_completion_recall_filters_degenerate_shard retargets its mock
#   from the no-longer-called ng.recall() to ng._harvest_associations(), with
#   the real "strength" key instead of "similarity".
# Why: cc_pattern_completion_recall() was rebuilt on spreading activation
#   (ng._harvest_associations -> graph.prime_and_propagate) instead of bare
#   ng.recall() cosine search -- see cc_ng_organism.py's own changelog and
#   docs/superpowers/specs/2026-07-07-cc-retrieval-enrichment-design.md.
#   Output shape and fail-soft behavior (the other 6 tests here) are unchanged.
# How: see inline comments at each adjusted assertion.
# -------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile, shutil
import pytest


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='cc_pattern_completion_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


def test_pattern_completion_recall_surfaces_aged_out_content(cc_ng):
    """The core outcome this build exists to produce: content that never fired
    through the SNN step pipeline (so SurfacingMonitor's recency queue never
    saw it -- run_conversational_dual_pass deposits via dual_record_outcome
    directly, not via on_message()/graph.step()) still surfaces for a later,
    semantically-related query via direct recall."""
    from cc_ng_organism import run_conversational_dual_pass, cc_pattern_completion_recall
    from ng_embed import embed

    text = "the deploy pipeline breaks whenever the redis cache is cold on first boot"
    state = {"last_forest_id": None}
    ok = run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, embed(text), state)
    assert ok is True

    # Confirm it genuinely is NOT in SurfacingMonitor's queue -- this is the
    # "aged out" (here: never-fired) scenario the pattern-completion path fixes.
    monitor = cc_ng._surfacing_monitor
    assert monitor is not None
    assert monitor.format_context() == ""

    # #358 rebuild note: recall is now substrate-native (cc_pattern_completion_recall
    # -> ng._harvest_associations -> graph.prime_and_propagate) -- a node only
    # surfaces if it actually FIRES, not merely if it is cosine-similar to the
    # query (that inversion is exactly what #358 removed). A freshly-deposited
    # conversational node is deliberately under novelty-dampening probation
    # (_cc_deposit_memory_node: threshold +0.2, intrinsic_excitability 0.3) so
    # real cosine similarity between this query and the deposited text (~0.78)
    # cannot cross even the un-boosted base firing threshold (0.85) -- probation
    # is intentional anti-spam design, not a regression. Lower the node's
    # threshold to make firing deterministic (same technique the #358 suite's
    # own test_recall_surfaces_synaptically_connected_not_cosine_similar uses for
    # its "assoc" node) so this test still proves its original point: previously
    # -deposited content that never ran through graph.step() is still reachable
    # via pattern-completion recall.
    cc_ng.graph.nodes[state["last_forest_id"]].threshold = 0.1

    results = cc_pattern_completion_recall(cc_ng, "why does the cache fail on cold boot", k=5)
    assert any(r["content"] == text for r in results)


def test_pattern_completion_recall_filters_degenerate_shard(cc_ng, monkeypatch):
    """resolve_surface_content's existing degenerate-fragment filter (bare
    stopword shard, under min_chars) must still apply -- a raw recall hit
    that resolves to nothing usable is dropped, not passed through.

    #358 rebuild note: cc_pattern_completion_recall no longer calls
    ng.recall() (bare cosine search) -- it calls ng._harvest_associations()
    (spreading activation). Mocking the old .recall() would now be dead code
    (the real call target moved), silently testing nothing; retarget the
    mock to the new call site and its real key ("strength", not
    "similarity") so this test still genuinely exercises the filter.
    """
    from cc_ng_organism import cc_pattern_completion_recall

    def fake_harvest(query, novelty=0.5):
        return [{"node_id": "nonexistent", "content": "want", "strength": 0.9}]
    monkeypatch.setattr(cc_ng, "_harvest_associations", fake_harvest)

    results = cc_pattern_completion_recall(cc_ng, "anything", k=5)
    assert results == []


def test_format_cc_recall_block_renders_active_recall_header():
    from cc_ng_organism import _format_cc_recall_block
    results = [{"node_id": "n1", "score": 0.87, "content": "example content"}]
    block = _format_cc_recall_block(results)
    assert block == "## Active Recall\nDirect memory retrieval for the current query:\n- [0.87] example content"


def test_format_cc_recall_block_empty_when_no_results():
    from cc_ng_organism import _format_cc_recall_block
    assert _format_cc_recall_block([]) == ""


def test_gate_pattern_completion_first_touch_returns_true_and_records():
    from cc_ng_organism import gate_pattern_completion
    cache = {}
    assert gate_pattern_completion(cache, "/a/b.py", 1000.0) is True
    assert cache["/a/b.py"] == 1000.0


def test_gate_pattern_completion_repeat_touch_within_ttl_returns_false():
    from cc_ng_organism import gate_pattern_completion
    cache = {"/a/b.py": 1000.0}
    assert gate_pattern_completion(cache, "/a/b.py", 1000.0 + 60.0) is False
    assert cache["/a/b.py"] == 1000.0  # untouched


def test_gate_pattern_completion_after_ttl_returns_true_and_refreshes():
    from cc_ng_organism import gate_pattern_completion, PATTERN_COMPLETION_FILE_TTL
    cache = {"/a/b.py": 1000.0}
    later = 1000.0 + PATTERN_COMPLETION_FILE_TTL + 1.0
    assert gate_pattern_completion(cache, "/a/b.py", later) is True
    assert cache["/a/b.py"] == later


def test_gate_pattern_completion_different_files_are_independent():
    from cc_ng_organism import gate_pattern_completion
    cache = {"/a/b.py": 1000.0}
    assert gate_pattern_completion(cache, "/c/d.py", 1000.0 + 1.0) is True
    assert cache["/c/d.py"] == 1000.0 + 1.0
    assert cache["/a/b.py"] == 1000.0  # unaffected
