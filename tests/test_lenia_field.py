# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Tests for FieldStore mmap double-buffered field state
# -------------------

"""Tests for lenia.field.FieldStore."""

import os
import tempfile

import numpy as np
import pytest

from lenia.field import FieldStore


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestFieldStore:
    def test_create_fresh(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=10, channel_count=4)
        assert fs.entity_count == 10
        assert fs.channel_count == 4
        assert fs.tick_counter == 0
        assert fs.total_energy() == 0.0

    def test_read_write_buffer_separation(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=5, channel_count=2)
        read = fs.read_buffer()
        write = fs.write_buffer()
        # They should be different views
        assert read is not write
        assert read.shape == (5, 2)
        assert write.shape == (5, 2)

    def test_inject_energy(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=5, channel_count=2)
        dist = np.array([0.7, 0.3])
        fs.inject(0, 1.0, dist)
        state = fs.read(0)
        assert abs(state[0] - 0.7) < 1e-10
        assert abs(state[1] - 0.3) < 1e-10

    def test_total_energy(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=3, channel_count=2)
        dist = np.array([0.5, 0.5])
        fs.inject(0, 2.0, dist)
        fs.inject(1, 3.0, dist)
        assert abs(fs.total_energy() - 5.0) < 1e-10
        assert abs(fs.total_energy(channel=0) - 2.5) < 1e-10

    def test_swap_buffers(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=3, channel_count=2)
        # Write to current buffer
        fs.inject(0, 1.0, np.array([1.0, 0.0]))
        val_before = fs.read(0, channel=0)
        assert abs(val_before - 1.0) < 1e-10

        # Write something different to the write buffer
        wb = fs.write_buffer()
        wb[0, 0] = 99.0

        # After swap, reads should see the write buffer
        fs.swap()
        val_after = fs.read(0, channel=0)
        assert abs(val_after - 99.0) < 1e-10

    def test_tick_increments(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=2, channel_count=1)
        assert fs.tick_counter == 0
        fs.tick()
        assert fs.tick_counter == 1
        fs.tick()
        assert fs.tick_counter == 2

    def test_persistence_across_reload(self, tmp_dir):
        # Create and populate
        fs = FieldStore(tmp_dir, entity_count=3, channel_count=2)
        fs.inject(1, 5.0, np.array([0.6, 0.4]))
        fs.tick()
        fs.tick()
        fs.close()

        # Reload
        fs2 = FieldStore(tmp_dir, entity_count=3, channel_count=2)
        assert fs2.tick_counter == 2
        val = fs2.read(1, channel=0)
        assert abs(val - 3.0) < 1e-10  # 5.0 * 0.6

    def test_resize_entities_grow(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=3, channel_count=2)
        fs.inject(0, 1.0, np.array([1.0, 0.0]))
        fs.resize_entities(5)
        assert fs.entity_count == 5
        # Original data preserved
        val = fs.read(0, channel=0)
        assert abs(val - 1.0) < 1e-10
        # New entities are zero
        assert abs(fs.read(3, channel=0)) < 1e-10

    def test_resize_channels_grow(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=3, channel_count=2)
        fs.inject(0, 1.0, np.array([0.5, 0.5]))
        fs.resize_channels(4)
        assert fs.channel_count == 4
        # Original channels preserved
        val = fs.read(0, channel=0)
        assert abs(val - 0.5) < 1e-10
        # New channels are zero
        assert abs(fs.read(0, channel=2)) < 1e-10

    def test_snapshot_restore(self, tmp_dir):
        fs = FieldStore(tmp_dir, entity_count=3, channel_count=2)
        fs.inject(0, 7.0, np.array([1.0, 0.0]))
        fs.snapshot("test_snap")

        # Modify state
        fs.inject(0, 100.0, np.array([1.0, 0.0]))
        assert fs.read(0, channel=0) > 50.0

        # Restore
        fs.restore("test_snap")
        val = fs.read(0, channel=0)
        assert abs(val - 7.0) < 1e-10
