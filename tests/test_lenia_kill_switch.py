# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Tests for kill switch — enable/disable, energy watchdog, persistence
# -------------------

"""Tests for lenia.kill_switch.KillSwitch."""

import tempfile

import pytest

from lenia.config import LeniaConfig
from lenia.kill_switch import KillSwitch


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class MockEngine:
    def __init__(self):
        self.running = False

    def start(self, **kwargs):
        self.running = True

    def stop(self, **kwargs):
        self.running = False


class MockBridge:
    def __init__(self):
        self.connected = False

    def connect(self, graph=None):
        self.connected = True

    def disconnect(self):
        self.connected = False


class TestKillSwitch:
    def test_starts_disabled(self, tmp_dir):
        ks = KillSwitch(LeniaConfig(), tmp_dir)
        assert not ks.enabled

    def test_enable_disable(self, tmp_dir):
        ks = KillSwitch(LeniaConfig(), tmp_dir)
        engine = MockEngine()
        bridge = MockBridge()
        ks.set_components(engine, bridge)

        ks.enable(graph="fake_graph")
        assert ks.enabled
        assert engine.running
        assert bridge.connected

        ks.disable()
        assert not ks.enabled
        assert not engine.running
        assert not bridge.connected

    def test_persistence(self, tmp_dir):
        ks1 = KillSwitch(LeniaConfig(), tmp_dir)
        ks1.enable()
        assert ks1.enabled

        # Reload
        ks2 = KillSwitch(LeniaConfig(), tmp_dir)
        assert ks2.enabled

    def test_energy_watchdog_normal(self, tmp_dir):
        ks = KillSwitch(LeniaConfig(), tmp_dir)
        ks.enable()

        # Normal energy levels — should not trigger
        ks.check_energy(100.0, 0.0)
        ks.check_energy(100.5, 0.5)  # small injection
        assert ks.enabled

    def test_energy_watchdog_auto_disable(self, tmp_dir):
        config = LeniaConfig(critical_energy_deviation=0.5)
        ks = KillSwitch(config, tmp_dir)
        engine = MockEngine()
        bridge = MockBridge()
        ks.set_components(engine, bridge)
        ks.enable()

        # Initialize baseline
        ks.check_energy(100.0, 0.0)
        # Massive unexpected deviation
        ks.check_energy(200.0, 0.0)

        # Should auto-disable
        assert not ks.enabled
        assert not engine.running

    def test_disable_preserves_enabled_file(self, tmp_dir):
        """Kill switch writes 'False' to file — field state file is not touched."""
        ks = KillSwitch(LeniaConfig(), tmp_dir)
        ks.enable()
        ks.disable(reason="test")
        # The enabled file should read False
        enabled_file = ks._path
        assert enabled_file.read_text().strip() == "False"

    def test_double_enable_noop(self, tmp_dir):
        ks = KillSwitch(LeniaConfig(), tmp_dir)
        engine = MockEngine()
        ks.set_components(engine, MockBridge())
        ks.enable()
        ks.enable()  # should not error
        assert ks.enabled

    def test_double_disable_noop(self, tmp_dir):
        ks = KillSwitch(LeniaConfig(), tmp_dir)
        ks.disable()  # already disabled
        assert not ks.enabled
