# ---- Changelog ----
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — Resume watermark tests
# What: watermark save/load lifecycle, canonical processing order, resume-skip filter,
#   interruption -> resume -> full-rebuild equivalence, resume+growth coverage, and
#   backward compat with watermark-less caches (including the live pre-fix format).
# Why: the property that matters is "an interrupted multi-hour rebuild finishes correctly
#   across a restart," not "a field round-trips" — the equivalence test pins it end to end.
# How: real Graph/NeuroGraphSubstrate/DistanceCache (this suite's no-mocks convention);
#   interruption simulated by a substrate wrapper whose distance_vector raises after N
#   calls — mirroring how a real crash interrupts populate() from inside the loop.
# -------------------
"""Tests for DistanceCache's resume watermark (interrupted-rebuild recovery)."""

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


def test_watermark_none_on_fresh_and_completed_cache(tmp_cache_path):
    g = _build_chain_graph(8)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    assert cache.watermark is None
    cache.populate(sub, checkpoint_interval_secs=1e-9,
                   on_checkpoint=lambda: cache.save(tmp_cache_path))
    # Completion clears the watermark; the final save must look complete.
    assert cache.watermark is None
    cache.save(tmp_cache_path)
    loaded = DistanceCache.load(tmp_cache_path)
    assert loaded is not None and loaded.watermark is None


def test_checkpoint_save_carries_watermark(tmp_cache_path):
    g = _build_chain_graph(8)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    seen = []

    def _cp():
        cache.save(tmp_cache_path)
        mid = DistanceCache.load(tmp_cache_path)
        seen.append(mid.watermark)

    cache.populate(sub, checkpoint_interval_secs=1e-9, on_checkpoint=_cp)
    assert seen, "expected at least one mid-loop checkpoint (idx=0 qualifies)"
    assert all(wm is not None for wm in seen), (
        "every mid-loop checkpoint must persist its resume point")


def test_canonical_processing_order_is_max_min_sorted():
    g = _build_chain_graph(10)
    sub = _RecordingSubstrate(NeuroGraphSubstrate(g, None))
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    cache.populate(sub)
    real = sub._real
    keys = [tuple(sorted((real.entity_index(a), real.entity_index(b)))[::-1])
            for a, b in sub.pairs]  # (max, min) per processed pair
    assert keys == sorted(keys), "pairs must be processed in (max, min) order"


def test_resume_skips_at_or_before_watermark():
    g = _build_chain_graph(10)
    real = NeuroGraphSubstrate(g, None)
    # First, discover the canonical pair list via a recording run.
    probe = _RecordingSubstrate(real)
    DistanceCache(real.entity_count(), entity_ids=real.entities()).populate(probe)
    canonical = [tuple(sorted((real.entity_index(a), real.entity_index(b))))
                 for a, b in probe.pairs]
    assert len(canonical) >= 6
    wm = canonical[len(canonical) // 2]  # a middle pair, (min, max) form

    resumed = _RecordingSubstrate(real)
    cache = DistanceCache(real.entity_count(), entity_ids=real.entities())
    cache.populate(resumed, resume_watermark=wm)
    done = [tuple(sorted((real.entity_index(a), real.entity_index(b))))
            for a, b in resumed.pairs]
    wm_key = (wm[1], wm[0])
    assert done, "resume must still process the remaining pairs"
    assert all((p[1], p[0]) > wm_key for p in done), (
        "no pair at or before the watermark may be recomputed")
    assert set(done) == {p for p in canonical if (p[1], p[0]) > wm_key}, (
        "resume must process EXACTLY the pairs after the watermark")


def test_interrupted_populate_resumes_to_full_equivalence(tmp_cache_path):
    """The end-to-end property: crash mid-rebuild, save, reload, resume ->
    per-pair identical to an uninterrupted full rebuild."""
    g = _build_chain_graph(12)
    real = NeuroGraphSubstrate(g, None)

    cache = DistanceCache(real.entity_count(), entity_ids=real.entities())
    inter = _InterruptingSubstrate(real, allow=5)
    with pytest.raises(RuntimeError):
        cache.populate(inter, checkpoint_interval_secs=1e-9,
                       on_checkpoint=lambda: cache.save(tmp_cache_path))
    # Mimic neurograph_rpc's caught-exception path: save whatever was computed.
    cache.save(tmp_cache_path)

    loaded = DistanceCache.load(tmp_cache_path)
    assert loaded is not None
    assert loaded.watermark is not None, (
        "an interrupted run's save must carry a resume point")
    loaded.populate(real, resume_watermark=loaded.watermark)
    assert loaded.watermark is None  # completion cleared it

    full = DistanceCache(real.entity_count(), entity_ids=real.entities())
    full.populate(real)
    for a, b in [("n0", "n1"), ("n3", "n4"), ("n5", "n7"), ("n9", "n11"), ("n0", "n2")]:
        ia, ib = real.entity_index(a), real.entity_index(b)
        assert np.allclose(loaded.get_distance_vector(ia, ib),
                           full.get_distance_vector(ia, ib)), (
            f"resumed cache diverges from full rebuild at ({a}, {b})")


def test_resume_after_growth_covers_new_entities(tmp_cache_path):
    g = _build_chain_graph(10)
    old_sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(old_sub.entity_count(), entity_ids=old_sub.entities())
    inter = _InterruptingSubstrate(old_sub, allow=4)
    with pytest.raises(RuntimeError):
        cache.populate(inter, checkpoint_interval_secs=1e-9,
                       on_checkpoint=lambda: cache.save(tmp_cache_path))
    cache.save(tmp_cache_path)

    # Graph grows before the restart (no new old-old synapses — that class
    # is the documented accepted blind spot, not under test).
    for i in range(10, 13):
        g.create_node(node_id=f"n{i}")
    for i in range(9, 12):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)

    loaded = DistanceCache.load(tmp_cache_path)
    new_sub = NeuroGraphSubstrate(g, None, known_entity_order=loaded.entity_ids)
    loaded.resize(new_sub.entity_count(), new_entity_ids=new_sub.entities())
    loaded.populate(new_sub, resume_watermark=loaded.watermark)

    full = NeuroGraphSubstrate(g, None)
    full_cache = DistanceCache(full.entity_count(), entity_ids=full.entities())
    full_cache.populate(full)
    for a, b in [("n9", "n10"), ("n10", "n11"), ("n11", "n12"), ("n2", "n3")]:
        assert np.allclose(
            loaded.get_distance_vector(new_sub.entity_index(a), new_sub.entity_index(b)),
            full_cache.get_distance_vector(full.entity_index(a), full.entity_index(b)),
        ), f"resume+growth diverges from full rebuild at ({a}, {b})"


def test_load_backward_compat_without_watermark_key(tmp_cache_path):
    """Caches saved by pre-watermark code (including the LIVE VPS checkpoint
    format as of 2026-07-08) have no watermark key — must load as complete."""
    g = _build_chain_graph(6)
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    cache.populate(sub)
    cache.save(tmp_cache_path)   # watermark is None -> key omitted
    loaded = DistanceCache.load(tmp_cache_path)
    assert loaded is not None
    assert loaded.watermark is None
