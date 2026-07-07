# ---- Changelog ----
# [2026-07-06] Claude Code (Sonnet 5) — Periodic in-loop checkpointing tests
# What: test_populate_periodic_checkpoint_saves_a_loadable_cache_before_loop_completes,
#   test_populate_checkpoint_disabled_by_default_makes_no_checkpoint_calls.
# Why: proves populate()'s new checkpoint_interval_secs/on_checkpoint actually
#   produce a genuinely reloadable on-disk cache DURING the loop (not just
#   that a callback fired) -- the real-world property that matters is "a
#   process kill mid-populate now leaves something recoverable," not merely
#   "the callback was invoked."
# How: real DistanceCache/NeuroGraphSubstrate/Graph (matches this file's own
#   no-mocks convention). idx=0 always satisfies the `idx % 1000 == 0` gate
#   by construction, so a tiny checkpoint_interval_secs deterministically
#   fires the very first check without needing a 1000+-pair graph.
# [2026-07-05] CC (laptop) — Incremental Lenia distance-cache extension
# What: Tests for DistanceCache.populate(start_index=...)/resize(new_entity_ids=...)/
#   entity_ids save-load roundtrip, and NeuroGraphSubstrate's known_entity_order
#   append-stable indexing (+ its fallback when a known entity is missing).
# Why: neurograph_rpc.py's handle_bootstrap previously nuked and repopulated the
#   whole Lenia distance cache on ANY entity_count drift — an O(total synapses/
#   hyperedges) rebuild that took up to ~8 hours on Syl's live graph and was
#   never given enough uninterrupted restart time to reach save() again. Fixed
#   by extending the cache in place and only computing distances for genuinely
#   new entities. These tests verify the incremental path produces IDENTICAL
#   per-entity-pair distances to a full rebuild on the same final graph state,
#   and that old entries are left untouched (bit-identical) rather than
#   recomputed.
# How: real neuro_foundation.Graph + lenia.graph_substrate.NeuroGraphSubstrate
#   (no mocks) — the interaction between append-stable indexing and cache
#   position alignment is exactly what's under test, and a mock substrate
#   would hide any misalignment bug.
# -------------------

"""Tests for incremental DistanceCache extension + append-stable indexing."""

import tempfile

import numpy as np
import pytest

from lenia.graph_substrate import NeuroGraphSubstrate
from lenia.kernel import DistanceCache, NUM_DIST_COMPONENTS
from neuro_foundation import Graph


def _build_chain_graph(n_nodes: int) -> Graph:
    """A simple chain n0-n1-n2-...-n(n-1) with real synapses."""
    g = Graph()
    for i in range(n_nodes):
        g.create_node(node_id=f"n{i}")
    for i in range(n_nodes - 1):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)
    return g


@pytest.fixture
def tmp_cache_path():
    with tempfile.TemporaryDirectory() as d:
        yield f"{d}/distance_cache"


def test_distance_cache_save_load_roundtrip_preserves_entity_ids(tmp_cache_path):
    g = _build_chain_graph(10)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    cache.populate(sub)
    cache.save(tmp_cache_path)

    loaded = DistanceCache.load(tmp_cache_path)
    assert loaded is not None
    assert loaded.entity_count == sub.entity_count()
    assert loaded.entity_ids == sub.entities()


def test_distance_cache_load_missing_entity_ids_is_backward_compatible(tmp_cache_path):
    """Old on-disk caches saved before this fix have no entity_ids key at all."""
    g = _build_chain_graph(5)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count())  # no entity_ids — old-style
    cache.populate(sub)
    cache.save(tmp_cache_path)

    loaded = DistanceCache.load(tmp_cache_path)
    assert loaded is not None
    assert loaded.entity_ids is None


def test_resize_preserves_existing_entries():
    g = _build_chain_graph(10)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    cache.populate(sub)

    before = cache.get_distance_vector(0, 1).copy()
    assert np.any(before != 0.0), "sanity: adjacent chain nodes should have a nonzero distance"

    cache.resize(15, new_entity_ids=sub.entities() + ["extra_a", "extra_b", "extra_c", "extra_d", "extra_e"])

    after = cache.get_distance_vector(0, 1)
    assert np.array_equal(before, after), "resize must not alter existing entries"
    assert cache.entity_count == 15
    # New region is untouched (all zero) until populate() fills it in.
    new_region = cache.get_distance_vector(10, 11)
    assert np.all(new_region == 0.0)


def test_incremental_populate_matches_full_rebuild_and_preserves_old_entries(tmp_cache_path):
    """The core correctness property: extend+incremental-populate on a grown
    graph must produce the SAME per-entity-pair distances as a from-scratch
    full rebuild on that same final graph state -- and must leave every
    already-cached old-old pair bit-identical rather than recomputing it.
    """
    # 1. Build the "before" graph, populate + save a real cache for it.
    g = _build_chain_graph(10)
    old_sub = NeuroGraphSubstrate(g, None)
    old_n = old_sub.entity_count()
    old_cache = DistanceCache(old_n, entity_ids=old_sub.entities())
    old_cache.populate(old_sub)
    old_cache.save(tmp_cache_path)

    # Capture a specific old-old distance to verify it survives untouched.
    old_pair_before = old_cache.get_distance_vector(
        old_sub.entity_index("n3"), old_sub.entity_index("n4")
    ).copy()

    # 2. Grow the SAME graph (simulating normal operation between restarts):
    #    extend the chain with 5 more nodes.
    for i in range(10, 15):
        g.create_node(node_id=f"n{i}")
    for i in range(9, 14):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)

    # 3. Incremental path: load the saved cache, build a substrate with
    #    known_entity_order so old entities keep their positions, resize +
    #    populate only the new region.
    loaded_cache = DistanceCache.load(tmp_cache_path)
    incr_sub = NeuroGraphSubstrate(g, None, known_entity_order=loaded_cache.entity_ids)
    assert incr_sub.entity_count() == 15
    # Append-stability check: the first 10 entities keep their exact indices.
    for i in range(10):
        assert incr_sub.entity_index(f"n{i}") == old_sub.entity_index(f"n{i}")

    loaded_cache.resize(incr_sub.entity_count(), new_entity_ids=incr_sub.entities())
    loaded_cache.populate(incr_sub, start_index=old_n)

    # Old entry must be untouched, bit-identical.
    old_pair_after = loaded_cache.get_distance_vector(
        incr_sub.entity_index("n3"), incr_sub.entity_index("n4")
    )
    assert np.array_equal(old_pair_before, old_pair_after)

    # 4. Independent reference: full rebuild from scratch on the SAME final
    #    graph state (fresh substrate, no known_entity_order — its own
    #    alphabetical sort order will differ from incr_sub's, so compare by
    #    entity ID via each cache's own substrate, not by raw index).
    full_sub = NeuroGraphSubstrate(g, None)
    full_cache = DistanceCache(full_sub.entity_count(), entity_ids=full_sub.entities())
    full_cache.populate(full_sub)

    for a, b in [("n0", "n9"), ("n3", "n4"), ("n9", "n10"), ("n10", "n14"), ("n0", "n14")]:
        incr_vec = loaded_cache.get_distance_vector(
            incr_sub.entity_index(a), incr_sub.entity_index(b)
        )
        full_vec = full_cache.get_distance_vector(
            full_sub.entity_index(a), full_sub.entity_index(b)
        )
        assert np.allclose(incr_vec, full_vec), (
            f"incremental vs full rebuild mismatch for ({a}, {b}): "
            f"{incr_vec} vs {full_vec}"
        )


def test_known_entity_order_falls_back_to_full_sort_when_entity_missing():
    """If any known entity was removed from the graph (pruned), append-stable
    reuse would silently misalign the cache -- must fall back to a fresh sort
    instead of filtering the known list in place.
    """
    g = _build_chain_graph(5)
    sub = NeuroGraphSubstrate(g, None)
    known_order = sub.entities()
    assert "n2" in known_order

    g.remove_node("n2")
    # A genuinely new substrate view over the mutated graph, still handed the
    # stale known_order that includes the now-removed "n2".
    sub2 = NeuroGraphSubstrate(g, None, known_entity_order=known_order)

    # Must NOT have reused known_order positions (that would leave "n2" as a
    # dangling index or silently shift others) -- falls back to sorted().
    assert "n2" not in sub2.entities()
    assert sub2.entities() == sorted(n for n in known_order if n != "n2")


def test_known_entity_order_appends_new_entities_after_known_ones():
    g = _build_chain_graph(5)
    sub = NeuroGraphSubstrate(g, None)
    known_order = sub.entities()

    for i in range(5, 8):
        g.create_node(node_id=f"n{i}")

    sub2 = NeuroGraphSubstrate(g, None, known_entity_order=known_order)
    assert sub2.entities()[: len(known_order)] == known_order
    assert sub2.entities()[len(known_order):] == ["n5", "n6", "n7"]
    for eid in known_order:
        assert sub2.entity_index(eid) == known_order.index(eid)


def test_populate_periodic_checkpoint_saves_a_loadable_cache_before_loop_completes(tmp_cache_path):
    g = _build_chain_graph(10)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())

    checkpoint_calls = []

    def _on_checkpoint():
        cache.save(tmp_cache_path)
        loaded = DistanceCache.load(tmp_cache_path)
        checkpoint_calls.append(loaded is not None and loaded.entity_count == sub.entity_count())

    cache.populate(
        sub, checkpoint_interval_secs=1e-9, on_checkpoint=_on_checkpoint,
    )

    assert len(checkpoint_calls) >= 1, "expected at least one mid-loop checkpoint (idx=0 always qualifies)"
    assert all(checkpoint_calls), "every checkpoint must produce a genuinely loadable cache, not just fire a callback"


def test_populate_checkpoint_disabled_by_default_makes_no_checkpoint_calls():
    """checkpoint_interval_secs=0.0 (the default) must be a true no-op --
    existing callers that never pass these new kwargs get unchanged behavior."""
    g = _build_chain_graph(10)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())

    checkpoint_calls = []
    cache.populate(sub, on_checkpoint=lambda: checkpoint_calls.append(True))

    assert checkpoint_calls == []
    assert cache.populated is True  # the actual work still completed normally
