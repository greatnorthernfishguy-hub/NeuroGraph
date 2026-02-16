"""
Tests for NG-Lite — Lightweight NeuroGraph Learning Substrate.

Covers: node management, synapse learning, novelty detection,
persistence, bridge interface, capacity limits, stats, and
NGSaaSBridge integration.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ng_lite import (
    DEFAULT_CONFIG,
    NGBridge,
    NGLite,
    NGLiteNode,
    NGLiteSynapse,
    __version__,
)
from ng_bridge import NGSaaSBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ng():
    """Fresh NG-Lite instance with default config."""
    return NGLite(module_id="test_module")


@pytest.fixture
def small_ng():
    """NG-Lite with small capacity limits for pruning tests."""
    return NGLite(
        module_id="test_small",
        config={"max_nodes": 3, "max_synapses": 5},
    )


@pytest.fixture
def embedding():
    """A random 384-dim embedding, normalized."""
    rng = np.random.RandomState(42)
    emb = rng.randn(384)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def different_embedding():
    """A different random 384-dim embedding."""
    rng = np.random.RandomState(99)
    emb = rng.randn(384)
    return emb / np.linalg.norm(emb)


# ---------------------------------------------------------------------------
# Node Tests
# ---------------------------------------------------------------------------

class TestNodeManagement:

    def test_create_node(self, ng, embedding):
        node = ng.find_or_create_node(embedding)
        assert isinstance(node, NGLiteNode)
        assert node.activation_count == 1
        assert node.node_id != ""

    def test_find_existing_node(self, ng, embedding):
        node1 = ng.find_or_create_node(embedding)
        node2 = ng.find_or_create_node(embedding)
        # Same embedding should return same node
        assert node1.node_id == node2.node_id
        assert node2.activation_count == 2

    def test_similar_node_reuse(self, ng, embedding):
        """Slightly perturbed embedding should match existing node."""
        node1 = ng.find_or_create_node(embedding)
        # Add small noise
        rng = np.random.RandomState(123)
        noisy = embedding + rng.randn(384) * 0.01
        noisy = noisy / np.linalg.norm(noisy)
        node2 = ng.find_or_create_node(noisy)
        assert node1.node_id == node2.node_id

    def test_novel_embedding_creates_new_node(self, ng, embedding, different_embedding):
        node1 = ng.find_or_create_node(embedding)
        node2 = ng.find_or_create_node(different_embedding)
        assert node1.node_id != node2.node_id
        assert len(ng.nodes) == 2

    def test_node_capacity_pruning(self, small_ng):
        """When at max_nodes, least-used node is pruned."""
        embeddings = []
        for i in range(4):
            rng = np.random.RandomState(i * 100)
            emb = rng.randn(384)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        # Create 3 nodes (at capacity)
        for emb in embeddings[:3]:
            small_ng.find_or_create_node(emb)
        assert len(small_ng.nodes) == 3

        # Activate first node extra times so it's not LRU
        for _ in range(5):
            small_ng.find_or_create_node(embeddings[0])

        # Adding 4th should prune the least used
        small_ng.find_or_create_node(embeddings[3])
        assert len(small_ng.nodes) == 3  # Still at capacity

    def test_node_id_incremental(self, ng, embedding, different_embedding):
        """Node IDs follow n_1, n_2, ... pattern."""
        node1 = ng.find_or_create_node(embedding)
        node2 = ng.find_or_create_node(different_embedding)
        assert node1.node_id == "n_1"
        assert node2.node_id == "n_2"


# ---------------------------------------------------------------------------
# Synapse & Learning Tests
# ---------------------------------------------------------------------------

class TestLearning:

    def test_record_success_strengthens(self, ng, embedding):
        result = ng.record_outcome(embedding, "model_a", success=True)
        assert result["weight_after"] > 0.5  # Started at 0.5

    def test_record_failure_weakens(self, ng, embedding):
        result = ng.record_outcome(embedding, "model_a", success=False)
        assert result["weight_after"] < 0.5  # Started at 0.5

    def test_repeated_success_approaches_one(self, ng, embedding):
        for _ in range(50):
            result = ng.record_outcome(embedding, "model_a", success=True)
        assert result["weight_after"] > 0.95

    def test_repeated_failure_approaches_zero(self, ng, embedding):
        for _ in range(50):
            result = ng.record_outcome(embedding, "model_a", success=False)
        assert result["weight_after"] < 0.05

    def test_weight_bounded_zero_one(self, ng, embedding):
        # Extreme success
        for _ in range(100):
            ng.record_outcome(embedding, "model_a", success=True)
        syn_key = list(ng.synapses.keys())[0]
        assert 0.0 <= ng.synapses[syn_key].weight <= 1.0

        # Extreme failure
        for _ in range(100):
            ng.record_outcome(embedding, "model_a", success=False)
        assert 0.0 <= ng.synapses[syn_key].weight <= 1.0

    def test_multiple_targets_independent(self, ng, embedding):
        ng.record_outcome(embedding, "model_a", success=True)
        ng.record_outcome(embedding, "model_b", success=False)

        recs = ng.get_recommendations(embedding, top_k=5)
        # model_a should rank higher than model_b
        targets = [r[0] for r in recs]
        assert targets.index("model_a") < targets.index("model_b")

    def test_success_failure_counts(self, ng, embedding):
        ng.record_outcome(embedding, "target_x", success=True)
        ng.record_outcome(embedding, "target_x", success=True)
        ng.record_outcome(embedding, "target_x", success=False)

        key = list(ng.synapses.keys())[0]
        syn = ng.synapses[key]
        assert syn.success_count == 2
        assert syn.failure_count == 1
        assert syn.activation_count == 3

    def test_synapse_capacity_pruning(self, small_ng, embedding):
        """When at max_synapses, weakest synapse is pruned."""
        # Create 5 synapses (at capacity for small_ng)
        for i in range(5):
            small_ng.record_outcome(embedding, f"target_{i}", success=True)
        assert len(small_ng.synapses) == 5

        # Weaken one synapse dramatically
        small_ng.record_outcome(embedding, "target_0", success=False)
        small_ng.record_outcome(embedding, "target_0", success=False)
        small_ng.record_outcome(embedding, "target_0", success=False)

        # Adding 6th target should prune the weakest
        small_ng.record_outcome(embedding, "target_new", success=True)
        assert len(small_ng.synapses) <= 5


# ---------------------------------------------------------------------------
# Recommendations Tests
# ---------------------------------------------------------------------------

class TestRecommendations:

    def test_empty_recommendations(self, ng, embedding):
        recs = ng.get_recommendations(embedding)
        assert recs == []

    def test_recommendations_sorted_by_weight(self, ng, embedding):
        # Train: model_a = mostly success, model_b = mostly failure
        for _ in range(10):
            ng.record_outcome(embedding, "model_a", success=True)
        for _ in range(10):
            ng.record_outcome(embedding, "model_b", success=False)
        ng.record_outcome(embedding, "model_b", success=True)  # Keep above prune

        recs = ng.get_recommendations(embedding, top_k=5)
        assert len(recs) >= 2
        assert recs[0][0] == "model_a"
        assert recs[0][1] > recs[1][1]

    def test_top_k_limit(self, ng, embedding):
        for i in range(10):
            ng.record_outcome(embedding, f"model_{i}", success=True)
        recs = ng.get_recommendations(embedding, top_k=3)
        assert len(recs) == 3


# ---------------------------------------------------------------------------
# Novelty Detection Tests
# ---------------------------------------------------------------------------

class TestNovelty:

    def test_empty_graph_is_fully_novel(self, ng, embedding):
        novelty = ng.detect_novelty(embedding)
        assert novelty == 1.0

    def test_known_pattern_low_novelty(self, ng, embedding):
        ng.find_or_create_node(embedding)
        novelty = ng.detect_novelty(embedding)
        assert novelty < 0.1

    def test_different_pattern_high_novelty(self, ng, embedding, different_embedding):
        ng.find_or_create_node(embedding)
        novelty = ng.detect_novelty(different_embedding)
        assert novelty > 0.3

    def test_similar_pattern_moderate_novelty(self, ng, embedding):
        ng.find_or_create_node(embedding)
        # Add moderate noise
        rng = np.random.RandomState(777)
        noisy = embedding + rng.randn(384) * 0.3
        noisy = noisy / np.linalg.norm(noisy)
        novelty = ng.detect_novelty(noisy)
        assert 0.0 < novelty < 1.0


# ---------------------------------------------------------------------------
# Persistence Tests
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_load_roundtrip(self, ng, embedding):
        ng.record_outcome(embedding, "model_a", success=True)
        ng.record_outcome(embedding, "model_b", success=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)

            # Load into fresh instance
            ng2 = NGLite(module_id="restored")
            ng2.load(filepath)

            assert len(ng2.nodes) == len(ng.nodes)
            assert len(ng2.synapses) == len(ng.synapses)
            assert ng2._total_outcomes == ng._total_outcomes
            assert ng2._total_successes == ng._total_successes
            assert ng2.module_id == ng.module_id
        finally:
            os.unlink(filepath)

    def test_saved_weights_preserved(self, ng, embedding):
        for _ in range(20):
            ng.record_outcome(embedding, "model_a", success=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)
            ng2 = NGLite()
            ng2.load(filepath)

            # Weights should match
            for key in ng.synapses:
                assert key in ng2.synapses
                assert abs(ng.synapses[key].weight - ng2.synapses[key].weight) < 1e-10
        finally:
            os.unlink(filepath)

    def test_save_file_is_valid_json(self, ng, embedding):
        ng.record_outcome(embedding, "test", success=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert "version" in data
            assert "nodes" in data
            assert "synapses" in data
            assert data["module_id"] == "test_module"
        finally:
            os.unlink(filepath)

    def test_synapse_key_serialization(self, ng, embedding):
        """Synapse tuple keys serialize as pipe-delimited strings."""
        ng.record_outcome(embedding, "target_a", success=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)
            with open(filepath) as f:
                data = json.load(f)

            # Synapse keys should be "source|target" format
            for key in data["synapses"]:
                assert "|" in key
                parts = key.split("|", 1)
                assert len(parts) == 2
                assert parts[0].startswith("n_")
        finally:
            os.unlink(filepath)

    def test_load_forward_compatible(self, ng):
        """Loading a state with missing new config keys should use defaults."""
        state = {
            "version": "0.1.0",
            "module_id": "old_module",
            "config": {"max_nodes": 500},
            "nodes": {},
            "synapses": {},
            "counters": {},
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(state, f)
            filepath = f.name

        try:
            ng.load(filepath)
            assert ng.config["max_nodes"] == 500  # Overridden
            assert ng.config["max_synapses"] == DEFAULT_CONFIG["max_synapses"]
        finally:
            os.unlink(filepath)

    def test_counters_preserved(self, ng, embedding):
        """Node ID counter survives save/load."""
        ng.record_outcome(embedding, "a", success=True)
        assert ng._node_id_counter == 1

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)
            ng2 = NGLite()
            ng2.load(filepath)
            assert ng2._node_id_counter == 1
        finally:
            os.unlink(filepath)


# ---------------------------------------------------------------------------
# Bridge Tests
# ---------------------------------------------------------------------------

class MockBridge(NGBridge):
    """Test bridge that records calls and returns canned responses."""

    def __init__(self, connected=True):
        self._connected = connected
        self.calls = []

    def is_connected(self) -> bool:
        return self._connected

    def record_outcome(self, embedding, target_id, success, module_id, metadata=None):
        self.calls.append(("record_outcome", target_id, success))
        return {"cross_module": True, "insight": "test_insight"}

    def get_recommendations(self, embedding, module_id, top_k=3):
        self.calls.append(("get_recommendations", module_id, top_k))
        return [("bridge_model", 0.95, "cross-module recommendation")]

    def detect_novelty(self, embedding, module_id):
        self.calls.append(("detect_novelty", module_id))
        return 0.42

    def sync_state(self, local_state, module_id):
        self.calls.append(("sync_state", module_id))
        return {"synced": True}


class TestBridge:

    def test_bridge_record_outcome(self, embedding):
        bridge = MockBridge()
        ng = NGLite(module_id="test", bridge=bridge)
        result = ng.record_outcome(embedding, "model_a", success=True)
        assert "bridge_response" in result
        assert result["bridge_response"]["cross_module"] is True
        assert len(bridge.calls) == 1

    def test_bridge_recommendations(self, embedding):
        bridge = MockBridge()
        ng = NGLite(module_id="test", bridge=bridge)
        recs = ng.get_recommendations(embedding, top_k=3)
        assert recs[0] == ("bridge_model", 0.95)

    def test_bridge_novelty(self, embedding):
        bridge = MockBridge()
        ng = NGLite(module_id="test", bridge=bridge)
        novelty = ng.detect_novelty(embedding)
        assert novelty == 0.42

    def test_bridge_sync(self, embedding):
        bridge = MockBridge()
        ng = NGLite(module_id="test", bridge=bridge)
        ng.record_outcome(embedding, "x", success=True)
        result = ng.sync_with_bridge()
        assert result == {"synced": True}

    def test_disconnected_bridge_falls_back(self, embedding):
        bridge = MockBridge(connected=False)
        ng = NGLite(module_id="test", bridge=bridge)
        ng.record_outcome(embedding, "model_a", success=True)
        # Should use local learning, not bridge
        recs = ng.get_recommendations(embedding)
        assert len(bridge.calls) == 0  # Bridge never called
        assert recs[0][0] == "model_a"

    def test_connect_disconnect(self, ng, embedding):
        bridge = MockBridge()
        ng.record_outcome(embedding, "local_model", success=True)

        # Connect bridge
        ng.connect_bridge(bridge)
        recs = ng.get_recommendations(embedding)
        assert recs[0][0] == "bridge_model"

        # Disconnect — should fall back to local
        ng.disconnect_bridge()
        recs = ng.get_recommendations(embedding)
        assert recs[0][0] == "local_model"

    def test_bridge_failure_falls_back(self, embedding):
        """If bridge raises an exception, fall back to local gracefully."""

        class FailingBridge(NGBridge):
            def is_connected(self):
                return True

            def record_outcome(self, *args, **kwargs):
                raise ConnectionError("SaaS unavailable")

            def get_recommendations(self, *args, **kwargs):
                raise ConnectionError("SaaS unavailable")

            def detect_novelty(self, *args, **kwargs):
                raise ConnectionError("SaaS unavailable")

            def sync_state(self, *args, **kwargs):
                raise ConnectionError("SaaS unavailable")

        ng = NGLite(module_id="test", bridge=FailingBridge())
        # Should not raise — falls back to local
        ng.record_outcome(embedding, "fallback_model", success=True)
        recs = ng.get_recommendations(embedding)
        assert recs[0][0] == "fallback_model"
        novelty = ng.detect_novelty(embedding)
        assert isinstance(novelty, float)


# ---------------------------------------------------------------------------
# Stats Tests
# ---------------------------------------------------------------------------

class TestStats:

    def test_initial_stats(self, ng):
        stats = ng.get_stats()
        assert stats["node_count"] == 0
        assert stats["synapse_count"] == 0
        assert stats["total_outcomes"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["bridge_connected"] is False
        assert stats["version"] == __version__

    def test_stats_after_learning(self, ng, embedding):
        ng.record_outcome(embedding, "model_a", success=True)
        ng.record_outcome(embedding, "model_a", success=False)

        stats = ng.get_stats()
        assert stats["node_count"] == 1
        assert stats["synapse_count"] == 1
        assert stats["total_outcomes"] == 2
        assert stats["total_successes"] == 1
        assert stats["success_rate"] == 0.5

    def test_memory_estimate(self, ng, embedding):
        ng.record_outcome(embedding, "model_a", success=True)
        stats = ng.get_stats()
        assert stats["memory_estimate_bytes"] > 0


# ---------------------------------------------------------------------------
# NGSaaSBridge Tests
# ---------------------------------------------------------------------------

class TestNGSaaSBridge:

    def _make_mock_memory(self):
        """Create a mock NeuroGraphMemory for testing."""
        memory = MagicMock()
        memory.on_message.return_value = {
            "status": "ingested",
            "nodes_created": 2,
            "synapses_created": 1,
        }
        memory.recall.return_value = [
            {
                "content": "[test_mod] outcome: target=model_a success=True",
                "similarity": 0.85,
                "node_id": "uuid-123",
            }
        ]
        memory.stats.return_value = {
            "nodes": 100,
            "synapses": 200,
            "firing_rate": 0.05,
            "prediction_accuracy": 0.75,
        }
        memory.save.return_value = "/tmp/checkpoint.msgpack"
        return memory

    def test_is_connected(self):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        assert bridge.is_connected() is True

    def test_disconnect_reconnect(self):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        bridge.disconnect()
        assert bridge.is_connected() is False
        bridge.reconnect()
        assert bridge.is_connected() is True

    def test_record_outcome_forwards(self, embedding):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        result = bridge.record_outcome(
            embedding, "model_a", True, "test_mod"
        )
        assert result is not None
        assert result["cross_module"] is True
        assert result["graph_nodes"] == 100
        memory.on_message.assert_called_once()

    def test_record_outcome_disconnected(self, embedding):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        bridge.disconnect()
        result = bridge.record_outcome(
            embedding, "model_a", True, "test_mod"
        )
        assert result is None
        memory.on_message.assert_not_called()

    def test_get_recommendations(self, embedding):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        recs = bridge.get_recommendations(embedding, "test_mod", top_k=3)
        assert recs is not None
        assert len(recs) == 1
        assert recs[0][0] == "model_a"
        assert recs[0][1] == 0.85

    def test_get_recommendations_no_results(self, embedding):
        memory = self._make_mock_memory()
        memory.recall.return_value = []
        bridge = NGSaaSBridge(memory)
        recs = bridge.get_recommendations(embedding, "test_mod")
        assert recs is None

    def test_detect_novelty(self, embedding):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        novelty = bridge.detect_novelty(embedding, "test_mod")
        assert novelty is not None
        assert novelty == pytest.approx(0.15, abs=0.01)  # 1 - 0.85

    def test_detect_novelty_empty_graph(self, embedding):
        memory = self._make_mock_memory()
        memory.recall.return_value = []
        bridge = NGSaaSBridge(memory)
        novelty = bridge.detect_novelty(embedding, "test_mod")
        assert novelty == 1.0

    def test_sync_state(self, ng, embedding):
        ng.record_outcome(embedding, "model_a", success=True)
        for _ in range(10):
            ng.record_outcome(embedding, "model_a", success=True)

        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        local_state = ng._export_state()
        result = bridge.sync_state(local_state, "test_mod")

        assert result is not None
        assert result["synced"] is True
        assert result["sync_number"] == 1
        assert result["patterns_ingested"] >= 1
        memory.save.assert_called_once()

    def test_sync_disconnected(self, ng, embedding):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory)
        bridge.disconnect()
        result = bridge.sync_state({}, "test_mod")
        assert result is None

    def test_weight_normalization(self):
        memory = self._make_mock_memory()
        bridge = NGSaaSBridge(memory, max_weight=5.0)
        assert bridge._normalize_weight_to_full(0.5) == 2.5
        assert bridge._normalize_weight_to_full(1.0) == 5.0
        assert bridge._normalize_weight_to_full(0.0) == 0.0
        assert bridge._normalize_weight_to_lite(2.5) == 0.5
        assert bridge._normalize_weight_to_lite(5.0) == 1.0
        assert bridge._normalize_weight_to_lite(0.0) == 0.0

    def test_extract_target(self):
        assert NGSaaSBridge._extract_target_from_content(
            "[mod] outcome: target=model_a success=True"
        ) == "model_a"
        assert NGSaaSBridge._extract_target_from_content(
            "no target here"
        ) is None
        assert NGSaaSBridge._extract_target_from_content(
            "target=last_one"
        ) == "last_one"


# ---------------------------------------------------------------------------
# Integration: NG-Lite with NGSaaSBridge
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline_with_bridge(self, embedding):
        """NG-Lite + NGSaaSBridge end-to-end flow."""
        memory = MagicMock()
        memory.on_message.return_value = {
            "status": "ingested", "nodes_created": 1,
            "synapses_created": 0,
        }
        memory.recall.return_value = [
            {
                "content": "[test] outcome: target=api_model success=True",
                "similarity": 0.9,
                "node_id": "uuid-456",
            }
        ]
        memory.stats.return_value = {
            "nodes": 50, "synapses": 100,
            "firing_rate": 0.03, "prediction_accuracy": 0.8,
        }
        memory.save.return_value = "/tmp/test.msgpack"

        bridge = NGSaaSBridge(memory)
        ng = NGLite(module_id="integration_test", bridge=bridge)

        # Record outcome — should forward to bridge
        result = ng.record_outcome(embedding, "local_model", success=True)
        assert "bridge_response" in result
        assert result["bridge_response"]["cross_module"] is True

        # Get recommendations — should use bridge
        recs = ng.get_recommendations(embedding, top_k=5)
        assert recs[0][0] == "api_model"

        # Detect novelty — should use bridge
        novelty = ng.detect_novelty(embedding)
        assert novelty == pytest.approx(0.1, abs=0.01)

        # Sync
        sync_result = ng.sync_with_bridge()
        assert sync_result is not None
        assert sync_result["synced"] is True

    def test_bridge_disconnect_fallback(self, embedding):
        """Disconnecting bridge falls back to local learning."""
        memory = MagicMock()
        memory.on_message.return_value = {"status": "ingested", "nodes_created": 1}
        memory.stats.return_value = {"nodes": 10, "synapses": 5}

        bridge = NGSaaSBridge(memory)
        ng = NGLite(module_id="fallback_test", bridge=bridge)

        # Learn locally first
        ng.record_outcome(embedding, "local_model", success=True)

        # Disconnect bridge
        bridge.disconnect()

        # Recommendations should fall back to local
        recs = ng.get_recommendations(embedding, top_k=3)
        assert recs[0][0] == "local_model"

        # Sync should return None
        assert ng.sync_with_bridge() is None


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_vector(self, ng):
        """Zero vector should be handled gracefully."""
        zero = np.zeros(384)
        node = ng.find_or_create_node(zero)
        assert node is not None

    def test_single_dim_embedding(self, ng):
        """Very short embedding should work."""
        short = np.array([1.0])
        node = ng.find_or_create_node(short)
        assert node is not None

    def test_concurrent_targets_same_node(self, ng, embedding):
        """Same pattern mapping to many different targets."""
        for i in range(20):
            success = i % 3 != 0
            ng.record_outcome(embedding, f"target_{i}", success=success)
        assert len(ng.nodes) == 1
        assert len(ng.synapses) == 20

    def test_empty_string_module_id(self):
        ng = NGLite(module_id="")
        assert ng.module_id == ""
        stats = ng.get_stats()
        assert stats["module_id"] == ""

    def test_version_string(self):
        assert __version__ == "1.0.0"
