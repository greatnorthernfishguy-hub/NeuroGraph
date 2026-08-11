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


def test_double_interrupt_through_resume_path_no_double_count(tmp_cache_path):
    """Crash a full rebuild, resume, then crash the RESUME after it has
    checkpointed — and finish on a third pass. This is the only path that
    exercises the COO checkpoint-fold watermark advance: the second crash's
    save must carry the CHECKPOINTED pair, not the stale first-crash
    watermark. If the fold advanced the CSR but not the watermark, the third
    pass would recompute the (first_wm, checkpoint] region already resident in
    CSR and `existing + delta` would DOUBLE it — caught here by full per-pair
    equivalence with an uninterrupted rebuild."""
    g = _build_chain_graph(16)
    real = NeuroGraphSubstrate(g, None)

    # Pass 1: full rebuild (LIL), crash early.
    cache = DistanceCache(real.entity_count(), entity_ids=real.entities())
    with pytest.raises(RuntimeError):
        cache.populate(_InterruptingSubstrate(real, allow=4),
                       checkpoint_interval_secs=1e-9,
                       on_checkpoint=lambda: cache.save(tmp_cache_path))
    cache.save(tmp_cache_path)
    loaded1 = DistanceCache.load(tmp_cache_path)
    wm1 = loaded1.watermark
    assert wm1 is not None

    # Pass 2: resume (COO), let idx=0 checkpoint fold+advance, then crash.
    # allow>=2 guarantees the idx=0 checkpoint fires before the crash.
    with pytest.raises(RuntimeError):
        loaded1.populate(_InterruptingSubstrate(real, allow=3),
                         resume_watermark=wm1,
                         checkpoint_interval_secs=1e-9,
                         on_checkpoint=lambda: loaded1.save(tmp_cache_path))
    loaded1.save(tmp_cache_path)
    loaded2 = DistanceCache.load(tmp_cache_path)
    wm2 = loaded2.watermark
    assert wm2 is not None, "the resume's crash-save must carry a resume point"
    assert wm2 != wm1, (
        "the COO checkpoint fold must ADVANCE the watermark past the "
        "first-crash point — otherwise the resident CSR (folded forward) and "
        "the persisted watermark desync and the next resume double-counts")

    # Pass 3: resume from the checkpointed watermark to completion.
    loaded2.populate(real, resume_watermark=wm2)
    assert loaded2.watermark is None

    # Every pair must match an uninterrupted full rebuild exactly — no gaps,
    # no doubled values.
    full = DistanceCache(real.entity_count(), entity_ids=real.entities())
    full.populate(real)
    for c in range(NUM_DIST_COMPONENTS):
        got = loaded2.get_csr(c).toarray()
        exp = full.get_csr(c).toarray()
        assert np.allclose(got, exp), (
            f"component {c} diverges from full rebuild after double-interrupt "
            f"(max abs diff {np.abs(got - exp).max()})")


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


def test_resume_leaves_no_lil_resident(tmp_cache_path):
    """#137 residency proof: resuming an interrupted rebuild must NOT
    re-materialize the transient LIL write buffer over the loaded CSR (the
    ~5 GB balloon on Syl's live bootstrap). The resume path is incremental —
    it folds a COO delta into the resident CSR — so _components_lil must be
    None both on the freshly-loaded cache and after populate() returns."""
    g = _build_chain_graph(12)
    real = NeuroGraphSubstrate(g, None)

    cache = DistanceCache(real.entity_count(), entity_ids=real.entities())
    inter = _InterruptingSubstrate(real, allow=5)
    with pytest.raises(RuntimeError):
        cache.populate(inter, checkpoint_interval_secs=1e-9,
                       on_checkpoint=lambda: cache.save(tmp_cache_path))
    cache.save(tmp_cache_path)

    loaded = DistanceCache.load(tmp_cache_path)
    assert loaded is not None
    # Loaded from disk: CSR-resident, no LIL.
    assert loaded._components_lil is None, "load() must not hold a LIL buffer"
    assert loaded.watermark is not None

    loaded.populate(real, resume_watermark=loaded.watermark)
    # Resume completed via the COO-delta path — never tolil()'d the CSR.
    assert loaded._components_lil is None, (
        "resume must not leave the ~5GB LIL buffer resident (#137)")
    assert loaded.watermark is None
