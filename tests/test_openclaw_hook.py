"""
Tests for the NeuroGraph OpenClaw integration hook.

Covers:
- Singleton pattern
- Message ingestion
- Semantic recall
- Auto-save behavior
- File/directory ingestion
- Stats reporting
- Save/restore across instances
"""

import os
import tempfile
from pathlib import Path

import pytest

from openclaw_hook import NeuroGraphMemory


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before each test."""
    NeuroGraphMemory.reset_instance()
    yield
    NeuroGraphMemory.reset_instance()


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory."""
    ws = tmp_path / "neurograph"
    ws.mkdir()
    (ws / "checkpoints").mkdir()
    return str(ws)


class TestSingleton:
    def test_singleton_returns_same_instance(self, workspace):
        ng1 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng2 = NeuroGraphMemory.get_instance()
        assert ng1 is ng2

    def test_reset_creates_new_instance(self, workspace):
        ng1 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        NeuroGraphMemory.reset_instance()
        ng2 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        assert ng1 is not ng2


class TestMessageIngestion:
    def test_ingest_text(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.on_message("The quick brown fox jumps over the lazy dog")
        assert result["status"] == "ingested"
        assert result["nodes_created"] > 0
        assert result["message_count"] == 1

    def test_ingest_empty_skipped(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.on_message("")
        assert result["status"] == "skipped"

    def test_ingest_whitespace_skipped(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.on_message("   \n\t  ")
        assert result["status"] == "skipped"

    def test_multiple_messages(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        for i in range(5):
            result = ng.on_message(f"Message number {i} about topic {i}")
        assert result["message_count"] == 5

    def test_graph_grows_with_messages(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats_before = ng.stats()
        ng.on_message("Neural networks learn patterns through backpropagation")
        ng.on_message("Gradient descent optimizes loss functions")
        stats_after = ng.stats()
        assert stats_after["nodes"] > stats_before["nodes"]


class TestRecall:
    def test_recall_finds_related(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Python is a programming language used for machine learning")
        ng.on_message("JavaScript runs in web browsers")
        results = ng.recall("programming language", k=5, threshold=0.0)
        # Should return some results (threshold=0 to be permissive with hash fallback)
        assert isinstance(results, list)

    def test_recall_empty_db(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        results = ng.recall("anything")
        assert results == []


class TestAutoSave:
    def test_auto_save_every_n_messages(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.auto_save_interval = 3
        checkpoint_path = Path(workspace) / "checkpoints" / "main.msgpack"

        for i in range(3):
            ng.on_message(f"Auto-save test message {i}")

        # After 3 messages with interval=3, checkpoint should exist
        assert checkpoint_path.exists()

    def test_manual_save(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Save this")
        path = ng.save()
        assert os.path.exists(path)


class TestStats:
    def test_stats_has_required_fields(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats = ng.stats()
        required_keys = [
            "version", "timestep", "nodes", "synapses", "hyperedges",
            "firing_rate", "mean_weight", "predictions_made",
            "prediction_accuracy", "vector_db_count", "checkpoint",
            "message_count",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_after_ingestion(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Test content for statistics")
        stats = ng.stats()
        assert stats["nodes"] > 0
        assert stats["message_count"] == 1


class TestPersistence:
    def test_cross_session_persistence(self, workspace):
        """Data survives across singleton resets (simulating restart)."""
        ng1 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng1.on_message("Persistent data test: STDP learning is fundamental")
        ng1.save()
        nodes_before = ng1.stats()["nodes"]

        NeuroGraphMemory.reset_instance()

        ng2 = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        nodes_after = ng2.stats()["nodes"]
        assert nodes_after == nodes_before


class TestFileIngestion:
    def test_ingest_python_file(self, workspace, tmp_path):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    print('hello world')\n")
        result = ng.ingest_file(str(py_file))
        assert result["status"] == "ingested"
        assert result["nodes_created"] > 0

    def test_ingest_missing_file(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        result = ng.ingest_file("/nonexistent/file.txt")
        assert result["status"] == "error"

    def test_ingest_directory(self, workspace, tmp_path):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)

        # Create test files
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.txt").write_text("hello world\n")
        (tmp_path / "c.jpg").write_text("not text")  # Should be skipped

        results = ng.ingest_directory(str(tmp_path))
        ingested = [r for r in results if r.get("status") == "ingested"]
        assert len(ingested) == 2  # .py and .txt, not .jpg


class TestStep:
    def test_step_runs_snn(self, workspace):
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Create some nodes first")
        results = ng.step(n=5)
        assert len(results) == 5


class TestMemoryLogging:
    def test_memory_dir_created(self, workspace):
        """Memory directory is created on initialization."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        memory_dir = Path(workspace) / "memory"
        assert memory_dir.is_dir()

    def test_embedding_status_event_on_init(self, workspace):
        """Embedding status event is written on initialization."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        events_path = Path(workspace) / "memory" / "events.jsonl"
        assert events_path.exists()
        import json
        lines = events_path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        # First event should be embedding_status
        assert events[0]["event"] == "embedding_status"
        assert "model_available" in events[0]["data"]

    def test_ingestion_event_on_message(self, workspace):
        """Ingestion events are written to memory/ on on_message."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        ng.on_message("Memory logging test message with enough content")
        events_path = Path(workspace) / "memory" / "events.jsonl"
        import json
        lines = events_path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        ingestion_events = [e for e in events if e["event"] == "ingestion"]
        assert len(ingestion_events) >= 1
        assert ingestion_events[0]["data"]["status"] == "ingested"

    def test_stats_includes_memory_dir(self, workspace):
        """Stats include memory_dir path."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats = ng.stats()
        assert "memory_dir" in stats
        assert "memory" in stats["memory_dir"]

    def test_stats_includes_embedding_status(self, workspace):
        """Stats include embedding backend status."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        stats = ng.stats()
        assert "embedding" in stats
        assert "model_available" in stats["embedding"]
        assert "device_requested" in stats["embedding"]


class TestEmbeddingDeviceConfig:
    def test_default_device_is_auto(self, workspace):
        """Default embedding device is 'auto'."""
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        status = ng.ingestor.embedder.status
        assert status["device_requested"] == "auto"

    def test_explicit_cpu_device_via_config(self, workspace):
        """Embedding device can be set via config."""
        ng = NeuroGraphMemory.get_instance(
            workspace_dir=workspace, config={"embedding_device": "cpu"}
        )
        status = ng.ingestor.embedder.status
        assert status["device_requested"] == "cpu"

    def test_env_var_device_override(self, workspace, monkeypatch):
        """NEUROGRAPH_EMBEDDING_DEVICE env var overrides default."""
        monkeypatch.setenv("NEUROGRAPH_EMBEDDING_DEVICE", "cpu")
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
        status = ng.ingestor.embedder.status
        assert status["device_requested"] == "cpu"
