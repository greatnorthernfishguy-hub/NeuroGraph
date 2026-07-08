# ---- Changelog ----
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — #371 reconcile tests
# What: kernel-level tests for DistanceCache.reconcile_removals — compaction
#   preserves surviving distances exactly, watermark cut translation (direct,
#   fallback-on-pruned-endpoint, untranslatable->None), dirty-set remap,
#   save/load round-trip, no-op and empty-survivor contracts.
# Why: #371 — the removal-bail discarded hours-to-days of computed progress
#   (confirmed live on Syl 2026-07-08: ~1.79M pairs lost at one restart).
#   These tests pin the invariant that makes reconcile safe: monotone
#   compaction preserves the canonical (max, min) pair order, and a fallback
#   cut only ever moves DOWN (recompute-safe), never up (skip-unsafe).
# How: real Graph/NeuroGraphSubstrate/DistanceCache (no-mocks convention);
#   translation tests inject cache._watermark directly after a full populate —
#   the same field load() restores — because in-populate checkpoints in small
#   graphs always land on the first pair (idx % 1000 == 0 gate).
# -------------------
"""Tests for DistanceCache.reconcile_removals (#371)."""

import tempfile

import numpy as np
import pytest

from lenia.graph_substrate import NeuroGraphSubstrate
from lenia.kernel import DistanceCache, NUM_DIST_COMPONENTS
from neuro_foundation import Graph


def _build_chain_graph(n_nodes: int) -> Graph:
    g = Graph()
    for i in range(n_nodes):
        g.create_node(node_id=f"n{i}")
    for i in range(n_nodes - 1):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)
    return g


class _InterruptingSubstrate:
    """Delegates to a real substrate; raises after `allow` distance calls —
    the same shape as a real mid-loop crash inside populate()."""

    def __init__(self, real, allow: int):
        self._real = real
        self._allow = allow
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def distance_vector(self, a, b):
        if self.calls >= self._allow:
            raise RuntimeError("simulated mid-populate crash")
        self.calls += 1
        return self._real.distance_vector(a, b)


class _RecordingSubstrate:
    """Delegates to a real substrate, recording each pair's entity ids."""

    def __init__(self, real):
        self._real = real
        self.pairs = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def distance_vector(self, a, b):
        self.pairs.append((a, b))
        return self._real.distance_vector(a, b)


@pytest.fixture
def tmp_cache_path():
    with tempfile.TemporaryDirectory() as d:
        yield f"{d}/distance_cache"


def _populated_cache(n_nodes: int):
    """Full populate over a chain graph. Returns (graph, substrate, cache,
    processed_pairs) where processed_pairs is the canonical-order list of
    (entity_id_a, entity_id_b) actually computed."""
    g = _build_chain_graph(n_nodes)
    real = NeuroGraphSubstrate(g, None)
    rec = _RecordingSubstrate(real)
    cache = DistanceCache(real.entity_count(), entity_ids=real.entities())
    cache.populate(rec)
    return g, real, cache, list(rec.pairs)


def test_reconcile_noop_when_nothing_removed():
    _, real, cache, _ = _populated_cache(6)
    before_ids = list(cache.entity_ids)
    live = set(before_ids)
    out = cache.reconcile_removals(live)
    assert out == before_ids
    assert cache.entity_ids == before_ids
    assert cache.entity_count == len(before_ids)
    assert cache.watermark is None


def test_reconcile_returns_none_without_entity_ids():
    cache = DistanceCache(4)  # no entity_ids — pre-2026-07 format
    assert cache.reconcile_removals({"a", "b"}) is None


def test_reconcile_returns_none_when_nothing_survives():
    _, _, cache, _ = _populated_cache(5)
    assert cache.reconcile_removals({"unrelated"}) is None


def test_reconcile_preserves_surviving_distances_exactly():
    _, real, cache, _ = _populated_cache(8)
    ids = list(cache.entity_ids)
    # Snapshot every nonzero surviving pair's distance vector, keyed by id.
    removed_id = "n3"
    survivors = [e for e in ids if e != removed_id]
    old_idx = {e: i for i, e in enumerate(ids)}
    want = {}
    for a_pos in range(len(survivors)):
        for b_pos in range(a_pos + 1, len(survivors)):
            a, b = survivors[a_pos], survivors[b_pos]
            vec = cache.get_distance_vector(old_idx[a], old_idx[b])
            if np.any(np.abs(vec) > 1e-15):
                want[(a, b)] = vec

    out = cache.reconcile_removals(set(survivors))
    assert out == survivors
    assert cache.entity_ids == survivors
    assert cache.entity_count == len(survivors)
    for c in range(NUM_DIST_COMPONENTS):
        assert cache._components_lil[c].shape == (len(survivors), len(survivors))
    new_idx = {e: i for i, e in enumerate(survivors)}
    for (a, b), vec in want.items():
        got = cache.get_distance_vector(new_idx[a], new_idx[b])
        np.testing.assert_allclose(got, vec)


def test_reconcile_watermark_direct_translation():
    _, real, cache, processed = _populated_cache(9)
    ids = list(cache.entity_ids)
    old_idx = {e: i for i, e in enumerate(ids)}
    # Pick a mid-list computed pair as the pretend-interrupted cut, with
    # both endpoints at indices ABOVE the entity we remove, so the
    # translation genuinely shifts indices.
    a, b = processed[len(processed) // 2]
    ia, ib = old_idx[a], old_idx[b]
    wm = (min(ia, ib), max(ia, ib))
    assert wm[0] > 1, "test setup: cut endpoints must sit above the removed index"
    cache._watermark = wm
    removed_id = ids[1]  # index 1 — below both endpoints
    survivors = [e for e in ids if e != removed_id]
    out = cache.reconcile_removals(set(survivors))
    assert out == survivors
    # Monotone compaction: indices above the removed one shift down by 1.
    assert cache.watermark == (wm[0] - 1, wm[1] - 1)


def test_reconcile_watermark_fallback_when_endpoint_pruned():
    _, real, cache, processed = _populated_cache(9)
    ids = list(cache.entity_ids)
    old_idx = {e: i for i, e in enumerate(ids)}
    k = len(processed) // 2
    a, b = processed[k]
    ia, ib = old_idx[a], old_idx[b]
    wm = (min(ia, ib), max(ia, ib))
    cache._watermark = wm
    # Remove the MAX endpoint of the cut pair — direct translation impossible.
    removed_id = ids[wm[1]]
    survivors = [e for e in ids if e != removed_id]

    # Expected fallback: greatest surviving computed pair at or below the
    # cut in canonical (max, min) order, then mapped to new indices.
    def key(p):
        i, j = old_idx[p[0]], old_idx[p[1]]
        return (max(i, j), min(i, j))

    wm_key = (wm[1], wm[0])
    candidates = [
        p for p in processed
        if p[0] != removed_id and p[1] != removed_id and key(p) <= wm_key
    ]
    assert candidates, "test setup: need at least one surviving computed pair below the cut"
    best = max(candidates, key=key)
    new_idx = {e: i for i, e in enumerate(survivors)}
    expected = tuple(sorted((new_idx[best[0]], new_idx[best[1]])))

    out = cache.reconcile_removals(set(survivors))
    assert out == survivors
    assert cache.watermark == expected


def test_reconcile_returns_none_when_cut_untranslatable():
    g = _build_chain_graph(8)
    real = NeuroGraphSubstrate(g, None)
    interrupting = _InterruptingSubstrate(real, allow=1)
    cache = DistanceCache(real.entity_count(), entity_ids=real.entities())
    with pytest.raises(RuntimeError):
        cache.populate(interrupting, checkpoint_interval_secs=1e-9,
                       on_checkpoint=lambda: None)
    wm = cache.watermark
    assert wm is not None, "checkpoint at idx=0 must have set the watermark"
    ids = list(cache.entity_ids)
    # Remove the watermark's max endpoint. Only ONE pair was ever computed
    # (the watermark pair itself), so no surviving computed pair <= cut
    # exists — reconcile must give up and return None.
    removed_id = ids[wm[1]]
    survivors = [e for e in ids if e != removed_id]
    assert cache.reconcile_removals(set(survivors)) is None


def test_reconcile_dirty_set_translated():
    _, real, cache, _ = _populated_cache(7)
    ids = list(cache.entity_ids)
    cache.mark_dirty(2, 5, 0)   # survives (translated)
    cache.mark_dirty(1, 4, 3)   # dies (endpoint 1 removed)
    removed_id = ids[1]
    survivors = [e for e in ids if e != removed_id]
    cache.reconcile_removals(set(survivors))
    # Old index 2 -> 1, 5 -> 4 after removing index 1; both orientations kept.
    assert (1, 4, 0) in cache._dirty and (4, 1, 0) in cache._dirty
    assert not any(c == 3 for (_, _, c) in cache._dirty)


def test_reconcile_save_load_roundtrip(tmp_cache_path):
    _, real, cache, processed = _populated_cache(9)
    ids = list(cache.entity_ids)
    old_idx = {e: i for i, e in enumerate(ids)}
    a, b = processed[len(processed) // 2]
    ia, ib = old_idx[a], old_idx[b]
    cache._watermark = (min(ia, ib), max(ia, ib))
    removed_id = ids[0]
    survivors = [e for e in ids if e != removed_id]
    out = cache.reconcile_removals(set(survivors))
    assert out == survivors
    wm_after = cache.watermark
    cache.save(tmp_cache_path)
    loaded = DistanceCache.load(tmp_cache_path)
    assert loaded is not None
    assert loaded.entity_ids == survivors
    assert loaded.entity_count == len(survivors)
    assert loaded.watermark == wm_after
    new_idx = {e: i for i, e in enumerate(survivors)}
    for a_pos in range(len(survivors)):
        for b_pos in range(a_pos + 1, len(survivors)):
            i, j = a_pos, b_pos
            np.testing.assert_allclose(
                loaded.get_distance_vector(i, j),
                cache.get_distance_vector(i, j),
            )
