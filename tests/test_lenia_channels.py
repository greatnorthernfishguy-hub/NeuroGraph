# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Tests for channel registry — seeding, birth, merge, death
# -------------------

"""Tests for lenia.channels.ChannelRegistry."""

import tempfile

import numpy as np
import pytest

from lenia.channels import ChannelRegistry
from lenia.config import LeniaConfig


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestChannelRegistry:
    def test_seed_four_channels(self, tmp_dir):
        config = LeniaConfig()
        reg = ChannelRegistry(config, tmp_dir)
        assert reg.count == 4
        names = [reg.get(cid).name for cid in reg.channel_ids]
        assert "norepinephrine" in names
        assert "dopamine" in names
        assert "cortisol" in names
        assert "acetylcholine" in names

    def test_persistence(self, tmp_dir):
        config = LeniaConfig()
        reg1 = ChannelRegistry(config, tmp_dir)
        assert reg1.count == 4
        # Update some state
        reg1.update_energy(0, 5.0)

        # Reload
        reg2 = ChannelRegistry(config, tmp_dir)
        assert reg2.count == 4
        assert reg2.get(0).total_energy == 5.0

    def test_birth_trigger(self, tmp_dir):
        config = LeniaConfig(
            residual_birth_threshold=0.01,
            residual_birth_window=5,
        )
        reg = ChannelRegistry(config, tmp_dir)
        assert reg.count == 4

        # Push high residuals
        for _ in range(5):
            result = reg.check_birth(0.1)

        # Should have triggered birth
        assert reg.count == 5

    def test_birth_no_trigger_below_threshold(self, tmp_dir):
        config = LeniaConfig(
            residual_birth_threshold=0.5,
            residual_birth_window=5,
        )
        reg = ChannelRegistry(config, tmp_dir)

        for _ in range(10):
            reg.check_birth(0.1)

        assert reg.count == 4  # no birth

    def test_death_trigger(self, tmp_dir):
        config = LeniaConfig(
            energy_death_threshold=0.01,
            energy_death_window=5,
        )
        reg = ChannelRegistry(config, tmp_dir)
        target_id = reg.channel_ids[3]  # acetylcholine

        # Push near-zero energy
        for _ in range(5):
            reg.update_energy(target_id, 0.0001)

        result = reg.check_death()
        assert result == target_id
        assert reg.count == 3

    def test_merge_trigger(self, tmp_dir):
        config = LeniaConfig(
            correlation_merge_threshold=0.8,
            correlation_merge_window=5,
        )
        reg = ChannelRegistry(config, tmp_dir)
        n_entities = 10

        # Create highly correlated field states for channels 0 and 1
        for _ in range(5):
            field = np.random.rand(n_entities, 4)
            # Make channels 0 and 1 nearly identical
            field[:, 1] = field[:, 0] * 1.01 + 0.001
            result = reg.check_merge(field)

        # Should have merged
        assert reg.count == 3

    def test_channel_index_mapping(self, tmp_dir):
        config = LeniaConfig()
        reg = ChannelRegistry(config, tmp_dir)
        ids = reg.channel_ids
        for i, cid in enumerate(ids):
            assert reg.channel_index(cid) == i

    def test_kernel_and_growth_access(self, tmp_dir):
        config = LeniaConfig()
        reg = ChannelRegistry(config, tmp_dir)
        cid = reg.channel_ids[0]
        ks = reg.kernel_shape(cid)
        gf = reg.growth_function(cid)
        assert "type" in ks
        assert "params" in ks
        assert "type" in gf
        assert "params" in gf
