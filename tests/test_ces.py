"""
Tests for the Cognitive Enhancement Suite (CES) — Phase 9.

Covers:
- CESConfig: defaults, overrides, JSON file loading
- StreamParser: chunking, embedding fallback, nudging, lifecycle
- ActivationPersistence: capture, save/restore, decay, bounds
- SurfacingMonitor: scoring, queue management, formatting
- CESMonitoring: health context, logger, dashboard
- Integration: CES wired into NeuroGraphMemory
"""

import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ces_config import (
    CESConfig,
    StreamingConfig,
    SurfacingConfig,
    PersistenceConfig,
    MonitoringConfig,
    load_ces_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@dataclass
class MockNode:
    """Minimal Node mock for tests."""
    node_id: str = "n1"
    voltage: float = 0.0
    threshold: float = 1.0
    resting_potential: float = 0.0
    refractory_remaining: int = 0
    refractory_period: int = 2
    last_spike_time: float = -math.inf
    firing_rate_ema: float = 0.0
    intrinsic_excitability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_inhibitory: bool = False


@dataclass
class MockHyperedge:
    """Minimal Hyperedge mock for tests."""
    hyperedge_id: str = "he1"
    member_node_ids: list = field(default_factory=list)
    member_weights: dict = field(default_factory=dict)
    threshold: float = 0.5
    is_archived: bool = False
    refractory_remaining: int = 0
    pattern_completion_strength: float = 0.3


@dataclass
class MockStepResult:
    """Minimal StepResult mock."""
    timestep: int = 0
    fired_node_ids: list = field(default_factory=list)
    fired_hyperedge_ids: list = field(default_factory=list)
    synapses_pruned: int = 0
    synapses_sprouted: int = 0


class MockVectorDB:
    """Minimal SimpleVectorDB mock."""

    def __init__(self):
        self._data: Dict[str, Dict] = {}

    def insert(self, entry_id, vector, metadata=None):
        self._data[entry_id] = {
            "vector": vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector,
            "content": (metadata or {}).get("content", ""),
            "metadata": metadata or {},
        }

    def search(self, query, k=10, threshold=0.0):
        results = []
        qnorm = query / np.linalg.norm(query) if np.linalg.norm(query) > 0 else query
        for eid, entry in self._data.items():
            sim = float(np.dot(qnorm, entry["vector"]))
            if sim >= threshold:
                results.append((eid, sim))
        results.sort(key=lambda x: -x[1])
        return results[:k]

    def get(self, entry_id):
        return self._data.get(entry_id)

    def count(self):
        return len(self._data)


class MockGraph:
    """Minimal Graph mock."""

    def __init__(self):
        self.nodes: Dict[str, MockNode] = {}
        self.hyperedges: Dict[str, MockHyperedge] = {}
        self.timestep: int = 0
        self.config: Dict[str, Any] = {}

    def step(self):
        self.timestep += 1
        return MockStepResult(timestep=self.timestep)


@pytest.fixture
def ces_config():
    """Default CES config."""
    return load_ces_config()


@pytest.fixture
def graph():
    """Mock graph with some nodes."""
    g = MockGraph()
    for i in range(5):
        nid = f"node_{i}"
        g.nodes[nid] = MockNode(node_id=nid, voltage=0.5 * i)
    return g


@pytest.fixture
def vector_db(graph):
    """Mock vector DB with entries matching graph nodes."""
    db = MockVectorDB()
    for nid in graph.nodes:
        vec = np.random.RandomState(hash(nid) % 2**31).randn(64).astype(np.float32)
        db.insert(nid, vec, {"content": f"Content for {nid}", "metadata": {"source": "test"}})
    return db


# ══════════════════════════════════════════════════════════════════════
# CESConfig Tests
# ══════════════════════════════════════════════════════════════════════


class TestCESConfigDefaults:
    def test_default_streaming(self):
        cfg = load_ces_config()
        assert cfg.streaming.ollama_model == "nomic-embed-text"
        assert cfg.streaming.chunk_size == 50
        assert cfg.streaming.overlap == 10
        assert cfg.streaming.nudge_strength == 0.15

    def test_default_surfacing(self):
        cfg = load_ces_config()
        assert cfg.surfacing.voltage_threshold == 0.6
        assert cfg.surfacing.max_surfaced == 5
        assert cfg.surfacing.decay_rate == 0.95

    def test_default_persistence(self):
        cfg = load_ces_config()
        assert cfg.persistence.sidecar_suffix == ".activations.json"
        assert cfg.persistence.decay_per_hour == 0.1
        assert cfg.persistence.max_entries == 10000

    def test_default_monitoring(self):
        cfg = load_ces_config()
        assert cfg.monitoring.http_port == 8847
        assert cfg.monitoring.http_enabled is True
        assert cfg.monitoring.backup_count == 5


class TestCESConfigOverrides:
    def test_override_streaming(self):
        cfg = load_ces_config({"streaming": {"chunk_size": 100, "overlap": 20}})
        assert cfg.streaming.chunk_size == 100
        assert cfg.streaming.overlap == 20
        # Other defaults preserved
        assert cfg.streaming.ollama_model == "nomic-embed-text"

    def test_override_surfacing(self):
        cfg = load_ces_config({"surfacing": {"max_surfaced": 20}})
        assert cfg.surfacing.max_surfaced == 20

    def test_override_multiple_sections(self):
        cfg = load_ces_config({
            "streaming": {"chunk_size": 75},
            "persistence": {"decay_per_hour": 0.2},
        })
        assert cfg.streaming.chunk_size == 75
        assert cfg.persistence.decay_per_hour == 0.2

    def test_unknown_keys_ignored(self):
        cfg = load_ces_config({"streaming": {"nonexistent_key": 42}})
        assert not hasattr(cfg.streaming, "nonexistent_key")


class TestCESConfigFile:
    def test_load_from_json_file(self, tmp_path):
        config_file = tmp_path / "ces.json"
        config_file.write_text(json.dumps({
            "streaming": {"chunk_size": 200},
            "monitoring": {"http_port": 9999},
        }))
        cfg = load_ces_config(config_path=str(config_file))
        assert cfg.streaming.chunk_size == 200
        assert cfg.monitoring.http_port == 9999

    def test_dict_overrides_win_over_file(self, tmp_path):
        config_file = tmp_path / "ces.json"
        config_file.write_text(json.dumps({
            "streaming": {"chunk_size": 200},
        }))
        cfg = load_ces_config(
            overrides={"streaming": {"chunk_size": 300}},
            config_path=str(config_file),
        )
        assert cfg.streaming.chunk_size == 300

    def test_missing_file_uses_defaults(self):
        cfg = load_ces_config(config_path="/nonexistent/path.json")
        assert cfg.streaming.chunk_size == 50


# ══════════════════════════════════════════════════════════════════════
# StreamParser Tests
# ══════════════════════════════════════════════════════════════════════


class TestStreamParserChunking:
    def test_basic_chunking(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        ces_config.streaming.chunk_size = 5
        ces_config.streaming.overlap = 2
        parser = StreamParser(graph, vector_db, ces_config)
        try:
            chunks = parser._chunk_text("one two three four five six seven eight")
            assert len(chunks) >= 2
            # First chunk should have 5 words
            assert len(chunks[0].split()) == 5
        finally:
            parser.stop()

    def test_empty_text(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            chunks = parser._chunk_text("")
            assert chunks == []
        finally:
            parser.stop()

    def test_short_text_single_chunk(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        ces_config.streaming.chunk_size = 50
        parser = StreamParser(graph, vector_db, ces_config)
        try:
            chunks = parser._chunk_text("hello world")
            assert len(chunks) == 1
            assert chunks[0] == "hello world"
        finally:
            parser.stop()

    def test_overlap_produces_overlapping_chunks(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        ces_config.streaming.chunk_size = 4
        ces_config.streaming.overlap = 2
        parser = StreamParser(graph, vector_db, ces_config)
        try:
            text = "a b c d e f g h"
            chunks = parser._chunk_text(text)
            assert len(chunks) >= 2
            # Check overlap: last 2 words of chunk 0 should be first 2 of chunk 1
            words0 = chunks[0].split()
            words1 = chunks[1].split()
            assert words0[-2:] == words1[:2]
        finally:
            parser.stop()


class TestStreamParserNudging:
    def test_nudge_increases_voltage(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            node = graph.nodes["node_0"]
            initial_voltage = node.voltage

            # Directly test nudging
            similar = [("node_0", 0.8)]
            parser._nudge_nodes(similar)

            expected = initial_voltage + 0.8 * ces_config.streaming.nudge_strength
            assert node.voltage == pytest.approx(expected)
            assert parser._nudges_applied == 1
        finally:
            parser.stop()

    def test_nudge_skips_refractory_nodes(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            node = graph.nodes["node_0"]
            node.refractory_remaining = 2
            initial_voltage = node.voltage

            parser._nudge_nodes([("node_0", 0.8)])
            assert node.voltage == initial_voltage  # No change
            assert parser._nudges_applied == 0
        finally:
            parser.stop()

    def test_nudge_skips_missing_nodes(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            parser._nudge_nodes([("nonexistent", 0.8)])
            assert parser._nudges_applied == 0
        finally:
            parser.stop()


class TestStreamParserLifecycle:
    def test_initial_stats(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            stats = parser.get_stats()
            assert stats["chunks_processed"] == 0
            assert stats["nudges_applied"] == 0
            assert stats["is_running"] is True
        finally:
            parser.stop()

    def test_pause_resume(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            assert parser.is_running is True
            parser.pause()
            assert parser.is_running is False
            parser.resume()
            assert parser.is_running is True
        finally:
            parser.stop()

    def test_stop(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        parser.stop()
        assert parser.is_running is False
        # Feed after stop should not crash
        parser.feed("test")

    def test_feed_processes_text(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        ces_config.streaming.chunk_size = 3
        ces_config.streaming.overlap = 0
        # Use fallback embedder since Ollama won't be available
        def fake_embed(text):
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(64).astype(np.float32)

        parser = StreamParser(
            graph, vector_db, ces_config, fallback_embedder=fake_embed
        )
        try:
            parser.feed("word1 word2 word3 word4 word5 word6")
            # Wait for processing
            time.sleep(1.0)
            assert parser._chunks_processed > 0
        finally:
            parser.stop()


class TestStreamParserEmbedding:
    def test_ollama_check_caches_result(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            # First check (will fail since no Ollama)
            result1 = parser._check_ollama()
            parser._ollama_last_check = time.time()

            # Second check should use cache
            result2 = parser._check_ollama()
            assert result1 == result2
        finally:
            parser.stop()

    def test_fallback_embedder_used_when_ollama_unavailable(
        self, graph, vector_db, ces_config
    ):
        from stream_parser import StreamParser

        called = {"count": 0}

        def fake_embed(text):
            called["count"] += 1
            return np.ones(64, dtype=np.float32)

        parser = StreamParser(
            graph, vector_db, ces_config, fallback_embedder=fake_embed
        )
        try:
            parser._ollama_available = False
            result = parser._embed_chunk("test text")
            assert result is not None
            assert called["count"] == 1
        finally:
            parser.stop()

    def test_no_embedding_when_no_fallback(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config, fallback_embedder=None)
        try:
            parser._ollama_available = False
            result = parser._embed_chunk("test text")
            assert result is None
        finally:
            parser.stop()


class TestStreamParserCompletions:
    def test_trigger_completions_fires(self, graph, vector_db, ces_config):
        from stream_parser import StreamParser

        parser = StreamParser(graph, vector_db, ces_config)
        try:
            # Set up a hyperedge with members that are active
            he = MockHyperedge(
                hyperedge_id="he1",
                member_node_ids=["node_0", "node_1", "node_2"],
                member_weights={"node_0": 1.0, "node_1": 1.0, "node_2": 1.0},
                threshold=0.5,
            )
            graph.hyperedges["he1"] = he

            # Activate enough members
            graph.nodes["node_0"].voltage = 1.0
            graph.nodes["node_1"].voltage = 1.0
            graph.nodes["node_2"].voltage = 0.0  # Inactive — will get completed

            parser._trigger_completions()
            assert parser._completions_triggered == 1
            # Inactive member should have been pre-charged
            assert graph.nodes["node_2"].voltage > 0
        finally:
            parser.stop()


# ══════════════════════════════════════════════════════════════════════
# ActivationPersistence Tests
# ══════════════════════════════════════════════════════════════════════


class TestActivationCapture:
    def test_capture_non_zero_voltages(self, graph, ces_config):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        snapshot = ap.capture(graph)
        # node_0 has voltage 0.0 (skipped), node_1-4 have positive voltages
        assert "node_0" not in snapshot
        assert "node_1" in snapshot
        assert snapshot["node_1"]["voltage"] == 0.5

    def test_capture_includes_excitability(self, graph, ces_config):
        from activation_persistence import ActivationPersistence

        graph.nodes["node_1"].intrinsic_excitability = 1.5
        ap = ActivationPersistence(ces_config)
        snapshot = ap.capture(graph)
        assert snapshot["node_1"]["excitability"] == 1.5

    def test_capture_bounded_by_max_entries(self, ces_config):
        from activation_persistence import ActivationPersistence

        ces_config.persistence.max_entries = 3
        g = MockGraph()
        for i in range(10):
            nid = f"n{i}"
            g.nodes[nid] = MockNode(node_id=nid, voltage=float(i + 1))

        ap = ActivationPersistence(ces_config)
        snapshot = ap.capture(g)
        assert len(snapshot) == 3
        # Should keep the 3 highest voltage nodes
        voltages = [s["voltage"] for s in snapshot.values()]
        assert max(voltages) == 10.0


class TestActivationSaveRestore:
    def test_save_creates_sidecar(self, graph, ces_config, tmp_path):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        ckpt = str(tmp_path / "main.msgpack")
        sidecar = ap.save(graph, ckpt)
        assert Path(sidecar).exists()
        assert sidecar.endswith(".activations.json")

    def test_restore_recovers_voltages(self, graph, ces_config, tmp_path):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        ckpt = str(tmp_path / "main.msgpack")

        # Set specific voltages
        graph.nodes["node_2"].voltage = 0.99
        ap.save(graph, ckpt)

        # Reset voltages
        for node in graph.nodes.values():
            node.voltage = 0.0

        # Restore (with ~0 elapsed time, so minimal decay)
        restored = ap.restore(graph, ckpt)
        assert restored > 0
        assert graph.nodes["node_2"].voltage == pytest.approx(0.99, abs=0.05)

    def test_restore_missing_sidecar_returns_zero(self, graph, ces_config, tmp_path):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        restored = ap.restore(graph, str(tmp_path / "nonexistent.msgpack"))
        assert restored == 0

    def test_restore_skips_deleted_nodes(self, ces_config, tmp_path):
        from activation_persistence import ActivationPersistence

        g = MockGraph()
        g.nodes["a"] = MockNode(node_id="a", voltage=1.0)
        g.nodes["b"] = MockNode(node_id="b", voltage=2.0)

        ap = ActivationPersistence(ces_config)
        ckpt = str(tmp_path / "main.msgpack")
        ap.save(g, ckpt)

        # Remove node "a" from graph
        del g.nodes["a"]
        for node in g.nodes.values():
            node.voltage = 0.0

        restored = ap.restore(g, ckpt)
        assert restored == 1  # Only "b" restored


class TestActivationDecay:
    def test_decay_reduces_voltage(self, ces_config):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        entries = {
            "n1": {"voltage": 1.0, "excitability": 1.0, "timestamp": 0},
        }

        decayed = ap._apply_decay(entries, elapsed_hours=1.0)
        assert decayed["n1"]["voltage"] < 1.0
        expected = 1.0 * (1.0 - ces_config.persistence.decay_per_hour) ** 1.0
        assert decayed["n1"]["voltage"] == pytest.approx(expected)

    def test_decay_prunes_below_threshold(self, ces_config):
        from activation_persistence import ActivationPersistence

        ces_config.persistence.min_activation = 0.5
        ap = ActivationPersistence(ces_config)

        entries = {
            "strong": {"voltage": 10.0, "timestamp": 0},
            "weak": {"voltage": 0.01, "timestamp": 0},
        }

        decayed = ap._apply_decay(entries, elapsed_hours=1.0)
        assert "strong" in decayed
        assert "weak" not in decayed

    def test_zero_elapsed_no_decay(self, ces_config):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        entries = {"n1": {"voltage": 1.0}}
        decayed = ap._apply_decay(entries, elapsed_hours=0.0)
        assert decayed["n1"]["voltage"] == 1.0


class TestActivationStats:
    def test_initial_stats(self, ces_config):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        stats = ap.get_stats()
        assert stats["entries_saved"] == 0
        assert stats["last_save_time"] is None

    def test_stats_after_save(self, graph, ces_config, tmp_path):
        from activation_persistence import ActivationPersistence

        ap = ActivationPersistence(ces_config)
        ckpt = str(tmp_path / "main.msgpack")
        ap.save(graph, ckpt)
        stats = ap.get_stats()
        assert stats["entries_saved"] > 0
        assert stats["last_save_time"] is not None


class TestActivationAutoSave:
    def test_auto_save_starts_and_stops(self, graph, ces_config, tmp_path):
        from activation_persistence import ActivationPersistence

        ces_config.persistence.auto_save_interval = 0.2
        ap = ActivationPersistence(ces_config)
        ckpt = str(tmp_path / "main.msgpack")

        ap.start_auto_save(graph, ckpt)
        assert ap._auto_save_timer is not None
        time.sleep(0.5)
        ap.stop_auto_save()
        assert ap._auto_save_timer is None
        # Verify sidecar was created
        assert Path(ckpt + ".activations.json").exists()


# ══════════════════════════════════════════════════════════════════════
# SurfacingMonitor Tests
# ══════════════════════════════════════════════════════════════════════


class TestSurfacingAfterStep:
    def test_fired_node_above_threshold_surfaced(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        ces_config.surfacing.voltage_threshold = 0.5
        ces_config.surfacing.min_confidence = 0.1
        monitor = SurfacingMonitor(graph, vector_db, ces_config)

        # Make node_3 fire with high voltage
        graph.nodes["node_3"].voltage = 2.0
        step_result = MockStepResult(fired_node_ids=["node_3"])
        monitor.after_step(step_result)

        surfaced = monitor.get_surfaced()
        assert len(surfaced) > 0
        assert surfaced[0]["node_id"] == "node_3"

    def test_fired_node_below_threshold_not_surfaced(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        ces_config.surfacing.voltage_threshold = 5.0
        monitor = SurfacingMonitor(graph, vector_db, ces_config)

        graph.nodes["node_1"].voltage = 0.5
        step_result = MockStepResult(fired_node_ids=["node_1"])
        monitor.after_step(step_result)

        surfaced = monitor.get_surfaced()
        assert len(surfaced) == 0

    def test_node_without_content_not_surfaced(self, graph, ces_config):
        from surfacing import SurfacingMonitor

        empty_db = MockVectorDB()  # No content stored
        ces_config.surfacing.voltage_threshold = 0.1
        ces_config.surfacing.min_confidence = 0.1
        monitor = SurfacingMonitor(graph, empty_db, ces_config)

        graph.nodes["node_3"].voltage = 2.0
        step_result = MockStepResult(fired_node_ids=["node_3"])
        monitor.after_step(step_result)

        surfaced = monitor.get_surfaced()
        assert len(surfaced) == 0


class TestSurfacingScoring:
    def test_score_includes_voltage(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        monitor = SurfacingMonitor(graph, vector_db, ces_config)
        graph.nodes["node_1"].voltage = 2.0
        score = monitor._score_node("node_1", graph.nodes["node_1"])
        assert score > 0

    def test_higher_voltage_higher_score(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        monitor = SurfacingMonitor(graph, vector_db, ces_config)
        graph.nodes["node_1"].voltage = 1.0
        graph.nodes["node_2"].voltage = 2.0
        s1 = monitor._score_node("node_1", graph.nodes["node_1"])
        s2 = monitor._score_node("node_2", graph.nodes["node_2"])
        assert s2 > s1

    def test_hyperedge_membership_boosts_score(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        monitor = SurfacingMonitor(graph, vector_db, ces_config)
        graph.nodes["node_1"].voltage = 1.0

        score_without = monitor._score_node("node_1", graph.nodes["node_1"])

        # Add node_1 to a hyperedge
        he = MockHyperedge(member_node_ids=["node_1"])
        graph.hyperedges["he1"] = he

        score_with = monitor._score_node("node_1", graph.nodes["node_1"])
        assert score_with > score_without


class TestSurfacingQueue:
    def test_queue_bounded(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        ces_config.surfacing.queue_capacity = 3
        ces_config.surfacing.voltage_threshold = 0.1
        ces_config.surfacing.min_confidence = 0.1
        monitor = SurfacingMonitor(graph, vector_db, ces_config)

        # Add many nodes
        for i in range(10):
            nid = f"extra_{i}"
            graph.nodes[nid] = MockNode(node_id=nid, voltage=float(i + 1))
            vec = np.random.RandomState(i).randn(64).astype(np.float32)
            vector_db.insert(nid, vec, {"content": f"Extra content {i}"})

        fired = [f"extra_{i}" for i in range(10)]
        step_result = MockStepResult(fired_node_ids=fired)
        monitor.after_step(step_result)

        assert len(monitor._queue) <= 3

    def test_decay_removes_weak_items(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        ces_config.surfacing.decay_rate = 0.1  # Aggressive decay
        ces_config.surfacing.min_confidence = 0.5
        ces_config.surfacing.voltage_threshold = 0.1
        monitor = SurfacingMonitor(graph, vector_db, ces_config)

        graph.nodes["node_3"].voltage = 2.0
        step_result = MockStepResult(fired_node_ids=["node_3"])
        monitor.after_step(step_result)
        assert len(monitor._queue) > 0

        # Many decay rounds should flush the queue
        for _ in range(20):
            monitor.after_step(MockStepResult())

        assert len(monitor._queue) == 0

    def test_clear_empties_queue(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        ces_config.surfacing.voltage_threshold = 0.1
        ces_config.surfacing.min_confidence = 0.1
        monitor = SurfacingMonitor(graph, vector_db, ces_config)

        graph.nodes["node_3"].voltage = 2.0
        monitor.after_step(MockStepResult(fired_node_ids=["node_3"]))
        assert len(monitor._queue) > 0

        monitor.clear()
        assert len(monitor._queue) == 0


class TestSurfacingFormatting:
    def test_format_context_block(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        ces_config.surfacing.voltage_threshold = 0.1
        ces_config.surfacing.min_confidence = 0.1
        monitor = SurfacingMonitor(graph, vector_db, ces_config)

        graph.nodes["node_3"].voltage = 2.0
        monitor.after_step(MockStepResult(fired_node_ids=["node_3"]))

        context = monitor.format_context()
        assert "[NeuroGraph Surfaced Knowledge]" in context
        assert "Content for node_3" in context
        assert "confidence:" in context

    def test_format_empty_returns_empty_string(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        monitor = SurfacingMonitor(graph, vector_db, ces_config)
        context = monitor.format_context()
        assert context == ""


class TestSurfacingStats:
    def test_initial_stats(self, graph, vector_db, ces_config):
        from surfacing import SurfacingMonitor

        monitor = SurfacingMonitor(graph, vector_db, ces_config)
        stats = monitor.get_stats()
        assert stats["total_surfaced"] == 0
        assert stats["queue_depth"] == 0
        assert stats["avg_score"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# CES Monitoring Tests
# ══════════════════════════════════════════════════════════════════════


class TestHealthContext:
    def test_health_context_includes_nodes(self):
        from ces_monitoring import health_context

        mock_memory = MagicMock()
        mock_memory.stats.return_value = {
            "nodes": 1234,
            "synapses": 5678,
            "prediction_accuracy": 0.89,
            "ces": {
                "stream_parser": {"is_running": True},
                "surfacing": {"queue_depth": 3},
            },
        }

        result = health_context(mock_memory)
        assert "1,234 nodes" in result
        assert "5,678 synapses" in result
        assert "89% prediction accuracy" in result
        assert "stream parser active" in result
        assert "3 concepts surfaced" in result

    def test_health_context_handles_error(self):
        from ces_monitoring import health_context

        mock_memory = MagicMock()
        mock_memory.stats.side_effect = RuntimeError("boom")

        result = health_context(mock_memory)
        assert "unavailable" in result


class TestCESLogger:
    def test_log_event_writes_file(self, tmp_path, ces_config):
        from ces_monitoring import CESLogger

        ces_config.monitoring.log_dir = str(tmp_path)
        logger_inst = CESLogger(ces_config)

        logger_inst.log_event("test_event", {"key": "value"})

        log_file = tmp_path / "ces.log"
        assert log_file.exists()
        content = log_file.read_text()
        parsed = json.loads(content.strip())
        assert parsed["event"] == "test_event"
        assert parsed["data"]["key"] == "value"


class TestCESMonitorCoordinator:
    def test_get_health(self, ces_config):
        from ces_monitoring import CESMonitor

        mock_memory = MagicMock()
        mock_memory.stats.return_value = {
            "nodes": 100,
            "synapses": 200,
            "prediction_accuracy": 0.5,
            "vector_db_count": 50,
        }
        ces_config.monitoring.http_enabled = False

        monitor = CESMonitor(mock_memory, ces_config)
        try:
            health = monitor.get_health()
            assert health["nodes"] == 100
            assert health["dashboard_running"] is False
        finally:
            monitor.stop()

    def test_health_context_method(self, ces_config):
        from ces_monitoring import CESMonitor

        mock_memory = MagicMock()
        mock_memory.stats.return_value = {"nodes": 42, "synapses": 100}
        ces_config.monitoring.http_enabled = False

        monitor = CESMonitor(mock_memory, ces_config)
        try:
            ctx = monitor.health_context()
            assert "42 nodes" in ctx
        finally:
            monitor.stop()

    def test_start_stop_lifecycle(self, ces_config):
        from ces_monitoring import CESMonitor

        mock_memory = MagicMock()
        mock_memory.stats.return_value = {"nodes": 0, "synapses": 0}
        ces_config.monitoring.http_enabled = False

        monitor = CESMonitor(mock_memory, ces_config)
        monitor.start()
        assert monitor._health_timer is not None
        monitor.stop()
        assert monitor._health_timer is None


class TestMonitoringDashboard:
    def test_dashboard_disabled(self, ces_config):
        from ces_monitoring import MonitoringDashboard

        ces_config.monitoring.http_enabled = False
        dashboard = MonitoringDashboard(ces_config)
        dashboard.start()
        assert not dashboard.is_running
        dashboard.stop()

    def test_dashboard_start_stop(self, ces_config):
        from ces_monitoring import MonitoringDashboard

        ces_config.monitoring.http_enabled = True
        ces_config.monitoring.http_port = 0  # OS-assigned port
        mock_memory = MagicMock()
        mock_memory.stats.return_value = {"nodes": 0}

        dashboard = MonitoringDashboard(ces_config, ng_memory=mock_memory)
        try:
            dashboard.start()
            # Port 0 may or may not work depending on OS, so just check it didn't crash
        finally:
            dashboard.stop()


# ══════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════


class TestCESIntegration:
    """Test CES wired into NeuroGraphMemory."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        yield
        NeuroGraphMemory.reset_instance()

    @pytest.fixture
    def workspace(self, tmp_path):
        ws = tmp_path / "neurograph"
        ws.mkdir()
        (ws / "checkpoints").mkdir()
        return str(ws)

    def test_ces_initialized(self, workspace):
        from openclaw_hook import NeuroGraphMemory

        ng = NeuroGraphMemory(
            workspace_dir=workspace,
            config={"ces": {"monitoring": {"http_enabled": False}}},
        )
        try:
            assert ng._ces_config is not None
            assert ng._stream_parser is not None
            assert ng._activation_persistence is not None
            assert ng._surfacing_monitor is not None
            assert ng._ces_monitor is not None
        finally:
            if ng._stream_parser:
                ng._stream_parser.stop()
            if ng._ces_monitor:
                ng._ces_monitor.stop()

    def test_ces_disabled(self, workspace):
        from openclaw_hook import NeuroGraphMemory

        ng = NeuroGraphMemory(
            workspace_dir=workspace,
            config={"ces": {"enabled": False}},
        )
        assert ng._ces_config is None
        assert ng._stream_parser is None

    def test_on_message_includes_ces_surfaced(self, workspace):
        from openclaw_hook import NeuroGraphMemory

        ng = NeuroGraphMemory(
            workspace_dir=workspace,
            config={"ces": {"monitoring": {"http_enabled": False}}},
        )
        try:
            result = ng.on_message("Testing CES integration with a message")
            assert "ces_surfaced" in result
        finally:
            if ng._stream_parser:
                ng._stream_parser.stop()
            if ng._ces_monitor:
                ng._ces_monitor.stop()

    def test_save_creates_activation_sidecar(self, workspace):
        from openclaw_hook import NeuroGraphMemory

        ng = NeuroGraphMemory(
            workspace_dir=workspace,
            config={"ces": {"monitoring": {"http_enabled": False}}},
        )
        try:
            ng.on_message("Some content to create nodes")
            ng.save()
            sidecar = Path(workspace) / "checkpoints" / "main.msgpack.activations.json"
            assert sidecar.exists()
        finally:
            if ng._stream_parser:
                ng._stream_parser.stop()
            if ng._ces_monitor:
                ng._ces_monitor.stop()

    def test_stats_includes_ces(self, workspace):
        from openclaw_hook import NeuroGraphMemory

        ng = NeuroGraphMemory(
            workspace_dir=workspace,
            config={"ces": {"monitoring": {"http_enabled": False}}},
        )
        try:
            stats = ng.stats()
            assert "ces" in stats
            assert "stream_parser" in stats["ces"]
            assert "surfacing" in stats["ces"]
            assert "persistence" in stats["ces"]
            assert "monitor" in stats["ces"]
        finally:
            if ng._stream_parser:
                ng._stream_parser.stop()
            if ng._ces_monitor:
                ng._ces_monitor.stop()
