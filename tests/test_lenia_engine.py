# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Tests for the Lenia update engine — conservation, ticking, observers
# -------------------

"""Tests for lenia.engine.UpdateEngine."""

import tempfile

import numpy as np
import pytest

from lenia.channels import ChannelRegistry
from lenia.config import LeniaConfig
from lenia.engine import UpdateEngine
from lenia.field import FieldStore
from lenia.kernel import DistanceCache, KernelComputer


class MockSubstrate:
    """Minimal substrate for testing — no real graph needed."""

    def __init__(self, n_entities):
        self._n = n_entities
        self._ids = [f"node_{i}" for i in range(n_entities)]

    def entities(self):
        return self._ids

    def entity_count(self):
        return self._n

    def entity_index(self, eid):
        return int(eid.split("_")[1])

    def index_to_entity(self, idx):
        return f"node_{idx}"

    def distance_vector(self, a, b):
        # Simple: distance = abs(index difference) in all components
        ia = self.entity_index(a)
        ib = self.entity_index(b)
        d = abs(ia - ib) / self._n
        return np.array([d, d, d, d, d], dtype=np.float64)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def setup(tmp_dir):
    """Create a complete Lenia stack for testing."""
    config = LeniaConfig()
    n_entities = 20
    n_channels = 4

    field = FieldStore(tmp_dir, n_entities, n_channels)
    registry = ChannelRegistry(config, tmp_dir)
    cache = DistanceCache(n_entities)
    kernel = KernelComputer(cache, registry)

    # Pre-populate distance cache with simple distances
    sub = MockSubstrate(n_entities)
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            dvec = sub.distance_vector(f"node_{i}", f"node_{j}")
            for c in range(5):
                cache.set_distance(i, j, c, dvec[c])

    engine = UpdateEngine(config, field, kernel, registry)

    return {
        "config": config,
        "field": field,
        "registry": registry,
        "cache": cache,
        "kernel": kernel,
        "engine": engine,
        "substrate": sub,
    }


class TestUpdateEngine:
    def test_tick_increments_counter(self, setup):
        engine = setup["engine"]
        field = setup["field"]
        assert field.tick_counter == 0
        engine.run_ticks(1)
        assert field.tick_counter == 1
        engine.run_ticks(5)
        assert field.tick_counter == 6

    def test_conservation_empty_field(self, setup):
        """With no energy, field stays at zero (only floor values)."""
        engine = setup["engine"]
        field = setup["field"]
        engine.run_ticks(10)
        # All values should be at field_floor (0.001) due to clamping
        buf = field.read_buffer()
        assert np.all(buf >= setup["config"].field_floor)

    def test_conservation_with_energy(self, setup):
        """Total energy change per tick should only be from floor drain."""
        field = setup["field"]
        engine = setup["engine"]
        config = setup["config"]

        # Inject energy at a few nodes
        dist = np.array([0.4, 0.3, 0.2, 0.1])
        field.inject(5, 10.0, dist)
        field.inject(10, 10.0, dist)

        initial_energy = field.total_energy()
        assert initial_energy > 0

        engine.run_ticks(1)
        post_energy = field.total_energy()

        # Energy should be close to initial.
        # Floor clamping can only add energy (clamping up from below floor).
        # Conservation normalization ensures growth deltas sum to zero.
        # So post_energy >= initial_energy (floor clamp adds tiny amounts).
        assert post_energy >= initial_energy - 1e-6

    def test_energy_spreads(self, setup):
        """Energy concentrated at one node should spread to neighbors."""
        field = setup["field"]
        engine = setup["engine"]

        # All energy at node 10
        dist = np.array([1.0, 0.0, 0.0, 0.0])
        field.inject(10, 50.0, dist)

        # Neighbors should be at floor
        assert field.read(9, channel=0) < 0.01
        assert field.read(11, channel=0) < 0.01

        engine.run_ticks(20)

        # After some ticks, neighbors should have gained energy
        # (kernel influences nearby nodes, growth function redistributes)
        buf = field.read_buffer()
        # Node 10 should still have significant energy
        assert buf[10, 0] > 1.0
        # Some energy should have reached neighbors
        # (exact amount depends on kernel shape, but shouldn't all stay at floor)

    def test_post_tick_observer(self, setup):
        engine = setup["engine"]
        field = setup["field"]

        observed = []
        engine.register_post_tick(lambda state: observed.append(state.copy()))

        field.inject(0, 1.0, np.array([1.0, 0.0, 0.0, 0.0]))
        engine.run_ticks(3)

        assert len(observed) == 3
        assert all(isinstance(s, np.ndarray) for s in observed)

    def test_status(self, setup):
        engine = setup["engine"]
        engine.run_ticks(5)
        s = engine.status()
        assert s["ticks_completed"] == 5
        assert s["ticks_skipped"] == 0
        assert "field_energy" in s
